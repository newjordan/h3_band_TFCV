"""Frame-exact planned-event versus H3 review sheet; never labels contact as true.

python -m h3_tools.piano_event_sheet RUN CENTER --radius 2 --video CANDIDATE --output sheet.jpg
CENTER is the delivered 24fps frame index. 'press' comes from the source plan,
not an inferred H3 fingertip. Visible output contact still needs annotation.
"""

import argparse
import json
import pathlib

import cv2
import numpy as np

from h3_tools import skeleton_piano as piano
from h3_tools.piano_clock_audit import decoded_frame


def sheet(run, center, radius, output, video=None):
    config = piano.RUNS[run]
    video = pathlib.Path(video) if video else piano.ROOT / run / f"{run}_final.mp4"
    reference = piano.ROOT / piano.src(run)
    source = json.loads(
        pathlib.Path(config.get("source_edit", piano.SHOW_EDIT)).read_text()
    )["frames"]
    rows, events = [], []
    for i in range(max(0, center - radius), min(config["keep"], center + radius + 1)):
        take = config["take_f0"] + i
        dense = config["f0"] + config["densify"] * i
        plan = source[take]
        plate_path = reference / "plate" / f"f_{dense:05d}.png"
        plate = cv2.imread(str(plate_path))
        if plate is None:
            raise ValueError(f"Missing reference frame {plate_path}")
        h3 = decoded_frame(video, i).astype(np.uint8)
        row = np.zeros((405, 1280, 3), np.uint8)
        row[53:, :640] = cv2.resize(plate, (640, 352))
        row[53:, 640:] = cv2.resize(h3, (640, 352))
        owners = (
            ", ".join(
                f"{note}:{finger}" for note, finger in plan.get("press", {}).items()
            )
            or "none"
        )
        title = f"FINAL {i}  t={i / 24:.6f}s | TAKE {take} | DENSE {dense} | PLAN ONLY: {owners}"
        cv2.putText(
            row,
            title,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        for text, x in [
            ("Reference: stock MPFB2 rig, NOT observed execution", 8),
            ("H3 output: contact/ownership UNLABELLED", 648),
        ]:
            cv2.putText(
                row,
                text,
                (x, 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (70, 210, 255),
                1,
                cv2.LINE_AA,
            )
        events.append(
            {
                "final_frame": i,
                "take_frame": take,
                "dense_frame": dense,
                "output_seconds": i / 24,
                "planned_note_finger_owners": plan.get("press", {}),
                "planned_key_depths": plan.get("keys", {}),
                "observed_contact": "UNLABELLED",
                "observed_note_identity": "UNLABELLED",
            }
        )
        rows.append(row)
    if not rows:
        raise ValueError("No requested output frames in shot")
    output = pathlib.Path(output)
    if not cv2.imwrite(str(output), np.vstack(rows)):
        raise OSError(f"Cannot write {output}")
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "run": run,
                "video": str(video),
                "inspected_source_dimensions": [1280, 704],
                "sheet_cell_dimensions": [640, 352],
                "events": events,
                "warning": "Plan labels are not detections; do not score missing/occluded fingertips as correct.",
            },
            indent=2,
        )
        + "\n"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", choices=sorted(piano.RUNS))
    parser.add_argument("center", type=int)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--video", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.radius < 0:
        parser.error("radius must be nonnegative")
    print(sheet(args.run, args.center, args.radius, args.output, args.video))


if __name__ == "__main__":
    main()
