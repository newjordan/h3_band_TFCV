"""Does an H3 render match its Blender plate, hand by hand? MediaPipe hand landmarks on both, frame by frame.

    ~/h3/mv_tools/.venv/bin/python h3_tools/handmatch.py PLATE.mp4 RENDER.mp4 [label]

Per frame: the plate's hands (grey blockout) and the render's hands are detected (21 landmarks each), paired by wrist
distance, and the mean landmark distance is divided by the plate hand's size (wrist -> middle MCP). Reports: share of
plate hands the render also shows as a hand (detectable), median / p90 normalised landmark error, and the error of
the fingertips alone (the part that hits the keys).
"""
import sys, subprocess, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision

MODEL = "/home/frosty40/h3/mv_tools/hand_landmarker.task"
TIPS = [4, 8, 12, 16, 20]


def frames(path):
    w, h = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                             "stream=width,height", "-of", "csv=p=0", path]).decode().split(","))
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), w, h


def detect(path):
    F, w, h = frames(path)
    opts = vision.HandLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=MODEL),
                                        running_mode=vision.RunningMode.VIDEO, num_hands=2,
                                        min_hand_detection_confidence=0.3, min_tracking_confidence=0.3)
    out = []
    with vision.HandLandmarker.create_from_options(opts) as det:
        for i, f in enumerate(F):
            r = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f)), int(i * 1000 / 24))
            out.append([np.array([[p.x * w, p.y * h] for p in lm]) for lm in r.hand_landmarks])
    return out


def main():
    plate, render = detect(sys.argv[1]), detect(sys.argv[2])
    n = min(len(plate), len(render))
    found = tot = 0
    err, tip = [], []
    for i in range(n):
        for P in plate[i]:
            tot += 1
            if not render[i]:
                continue
            R = min(render[i], key=lambda R: np.linalg.norm(R[0] - P[0]))
            size = np.linalg.norm(P[9] - P[0]) + 1e-6
            if np.linalg.norm(R[0] - P[0]) > 2.5 * size:
                continue
            found += 1
            err.append(float(np.mean(np.linalg.norm(R - P, axis=1)) / size))
            tip.append(float(np.mean(np.linalg.norm(R[TIPS] - P[TIPS], axis=1)) / size))
    lab = sys.argv[3] if len(sys.argv) > 3 else sys.argv[2].split("/")[-1]
    if not err:
        print(f"{lab}: plate hands {tot}, render shows none of them as hands"); return
    print(f"{lab}: render shows {found}/{tot} plate hands ({100 * found / max(tot, 1):.0f}%); landmark error median "
          f"{np.median(err):.3f} p90 {np.percentile(err, 90):.3f}; fingertips median {np.median(tip):.3f} (x hand size)")


if __name__ == "__main__":
    main()
