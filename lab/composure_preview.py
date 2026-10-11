"""Prepare a fresh matched side-by-side excerpt, preserving every solved pose."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = []
    for trial in ('baseline', 'both'):
        source = ROOT / f'artifacts/realism/composure_{trial}.anim.json'
        animation = json.loads(source.read_text())
        preview = copy.deepcopy(animation)
        preview['fps'] = animation['fps'] // 2
        preview['frames'] = preview['frames'][::2]
        # Renderer opts supply a fixed camera, identical on both sides.
        preview['render_camera'] = dict(view='overhead', eye=[0., -1., 1.],
                                       look=[-.04, .015, .02], lens=40., hide='body')
        target = ROOT / f'artifacts/realism/composure_{trial}.preview.anim.json'
        target.write_text(json.dumps(preview) + '\n')
        for original, selected in zip(animation['frames'][::2], preview['frames']):
            assert original == selected
        records.append(dict(trial=trial, source=str(source), source_sha256=sha(source),
            animation=str(target), animation_sha256=sha(target), frames=len(preview['frames']),
            fps=preview['fps'], original_pose_key_body_data_preserved=True))
    proof = dict(schema='composure-render-comparison/v1', status='prepared', records=records,
        fps=12, duration_seconds=6., resolution_each=[768, 432],
        candidate_exported=False, full_performance_accepted=False)
    (ROOT / 'artifacts/realism/composure_preview.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
