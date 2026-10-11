"""Check physical interference between the actual rendered moving key solids."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blockout.piano_geometry import scene_from_animation


def box_pair_margin(scene, first, second):
    """Largest separating-axis slack for two oriented solid boxes (metres)."""
    ra, rb = scene.rotations[first], scene.rotations[second]
    axes = np.vstack([ra.T, rb.T, np.cross(ra.T[:, None, :], rb.T[None, :, :]).reshape(-1, 3)])
    lengths = np.linalg.norm(axes, axis=1)
    axes = axes[lengths > 1e-12] / lengths[lengths > 1e-12, None]
    ca = scene.origins[first] + ra @ scene.centers[first]
    cb = scene.origins[second] + rb @ scene.centers[second]
    radius_a = np.abs(axes @ ra) @ scene.half[first]
    radius_b = np.abs(axes @ rb) @ scene.half[second]
    return float(np.max(np.abs(axes @ (cb - ca)) - radius_a - radius_b))


def interference(scene, tolerance=1e-5, include_case=False):
    keys = [i for i, spec in enumerate(scene.specs) if spec.pitch is not None]
    contacts = []
    for n, first in enumerate(keys):
        others = keys[n + 1:] + ([i for i, spec in enumerate(scene.specs) if spec.pitch is None] if include_case else [])
        for second in others:
            if np.any(scene.aabb_upper[first] < scene.aabb_lower[second] - tolerance) or np.any(scene.aabb_upper[second] < scene.aabb_lower[first] - tolerance):
                continue
            margin = box_pair_margin(scene, first, second)
            if margin < -tolerance:
                contacts.append(dict(first=scene.specs[first].name, second=scene.specs[second].name,
                                     minimum_translation_mm=-margin * 1000.))
    return contacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('animation', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--frames', default='23,24,25')
    parser.add_argument('--notched', action='store_true')
    parser.add_argument('--include-case', action='store_true')
    args = parser.parse_args()
    source = json.loads(args.animation.read_text())
    if args.notched:
        source = copy.deepcopy(source)
        source.setdefault('geometry', {})['key_shape'] = 'notched'
    records = [dict(frame='rest', contacts=interference(scene_from_animation(source, dict(keys={})), include_case=args.include_case))]
    for index in map(int, args.frames.split(',')):
        frame = source['frames'][index]
        records.append(dict(frame=index, time=frame['t'], contacts=interference(scene_from_animation(source, frame), include_case=args.include_case)))
    scope = 'sampled moving key/key solids' + (' and key/case solids' if args.include_case else '')
    report = dict(source=str(args.animation.resolve()), source_sha256=hashlib.sha256(args.animation.read_bytes()).hexdigest(),
                  geometry=source.get('geometry', {'key_shape': 'rectangular'}),
                  geometry_override=bool(args.notched), scope=scope,
                  exclusions=['case/case assembly', 'forces', 'continuous time'],
                  tolerance_mm=.01, notched=source.get('geometry', {}).get('key_shape') == 'notched', include_case=args.include_case,
                  clear=not any(r['contacts'] for r in records), records=records)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
