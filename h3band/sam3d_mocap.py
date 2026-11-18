#!/usr/bin/env python3
"""Video -> BVH body mocap with SAM 3D Body (Meta, SAM License), run on a local ComfyUI's built-in nodes.

  sam3d_mocap.py VIDEO OUT_PREFIX [--fps 23.976] [--window 7] [--no-hands] [--url http://127.0.0.1:8188]

Uploads VIDEO, runs LoadVideo -> SAM3DBody_Predict (one person, whole frame as the box) and writes two BVHs:
OUT_PREFIX.raw.bvh (per-frame predictions) and OUT_PREFIX.bvh (after SAM3DBody_Smooth, Savitzky-Golay over
--window frames). Needs ComfyUI with the SAM 3D Body nodes (core since Aug 2026) and
models/detection/sam_3d_body_dinov3_bf16.safetensors (huggingface.co/Comfy-Org/sam-3d-body).

The smoother runs across shot cuts: drop a few frames either side of each cut before measuring.
"""
import argparse, json, mimetypes, os, shutil, sys, time, urllib.parse, urllib.request, uuid

MODEL = "sam_3d_body_dinov3_bf16.safetensors"


def _post(url, data, ctype):
    req = urllib.request.Request(url, data=data, headers={"Content-Type": ctype})
    return json.load(urllib.request.urlopen(req))


def upload(base, path):
    b = uuid.uuid4().hex
    name = os.path.basename(path)
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: {mimetypes.guess_type(name)[0] or 'application/octet-stream'}\r\n\r\n").encode()
    body += open(path, "rb").read() + f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{b}--\r\n".encode()
    return _post(f"{base}/upload/image", body, f"multipart/form-data; boundary={b}")["name"]


def graph(video, fps, window, hands, prefix):
    bvh = {"format": "bvh", "format.units": "cm", "fps": fps, "camera_translation": "off", "track_index": 0}
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "SAM3DBody_Loader", "inputs": {"model_file": MODEL}},
        "4": {"class_type": "SAM3DBody_Predict", "inputs": {"sam3d_body_model": ["3", 0], "image": ["2", 0],
                                                            "run_hand_refinement": hands, "fov": 0.0, "batch_size": 32}},
        "5": {"class_type": "SAM3DBody_Smooth", "inputs": {"mhr_pose_data": ["4", 0], "strength": 1.0, "method": "savgol",
                                                           "window": window, "rotation_threshold_degrees": 30.0}},
        "6": {"class_type": "BuildPoseFile", "inputs": {"pose_data": ["4", 0], "sam3d_body_model": ["3", 0], **bvh}},
        "7": {"class_type": "SaveGLB", "inputs": {"mesh": ["6", 0], "filename_prefix": f"{prefix}_raw"}},
        "8": {"class_type": "BuildPoseFile", "inputs": {"pose_data": ["5", 0], "sam3d_body_model": ["3", 0], **bvh}},
        "9": {"class_type": "SaveGLB", "inputs": {"mesh": ["8", 0], "filename_prefix": f"{prefix}_smooth"}},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video"); ap.add_argument("out_prefix")
    ap.add_argument("--fps", type=float, default=23.976)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--no-hands", action="store_true", help="skip the hand refinement pass (faster)")
    ap.add_argument("--url", default=os.environ.get("H3B_COMFY_URL", "http://127.0.0.1:8188"))
    ap.add_argument("--output-dir", help="the ComfyUI output dir, to copy the BVHs from (default: download via /view)")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    name = upload(base, a.video)
    tag = "sam3d/" + uuid.uuid4().hex[:8]
    pid = _post(f"{base}/prompt", json.dumps({"prompt": graph(name, a.fps, a.window, not a.no_hands, tag)}).encode(),
                "application/json")["prompt_id"]
    print(f"queued {pid}", flush=True)
    t0 = time.time()
    while True:
        h = json.load(urllib.request.urlopen(f"{base}/history/{pid}")).get(pid)
        if h and h.get("status", {}).get("completed") is not None:
            break
        if h and h.get("status", {}).get("status_str") == "error":
            break
        time.sleep(5)
    st = h["status"]
    if st.get("status_str") != "success":
        sys.exit(f"failed: {json.dumps(st.get('messages'), indent=1)[:4000]}")
    print(f"done in {time.time() - t0:.0f} s")
    for node, suffix in (("7", ".raw.bvh"), ("9", ".bvh")):
        f = h["outputs"][node]["3d"][0] if "3d" in h["outputs"][node] else next(iter(h["outputs"][node].values()))[0]
        dst = a.out_prefix + suffix
        if a.output_dir:
            shutil.copy(os.path.join(a.output_dir, f.get("subfolder", ""), f["filename"]), dst)
        else:
            q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": "output"})
            open(dst, "wb").write(urllib.request.urlopen(f"{base}/view?{q}").read())
        print(f"wrote {dst}")


if __name__ == "__main__":
    main()
