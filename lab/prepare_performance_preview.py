"""Extract a timed diagnostic excerpt without changing any physical pose.

The existing complete v19 cache is an older, size-1 planner baseline. New
static contact experiments have not passed; they must not be spliced into it.
Only frame selection, local timestamps and preview metadata change here.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--first-frame', type=int, default=4800)
    parser.add_argument('--seconds', type=int, default=60)
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--legacy-hand-scale', type=float, default=1.)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    original = json.loads(source_bytes)
    fps = original['fps']
    assert fps % args.stride == 0 and args.stride > 0 and args.seconds > 0
    assert original.get('speed', 1.) == 1., 'Audio mapping requires real-time source'
    scale = original.get('hand_scale', args.legacy_hand_scale)
    assert scale == args.legacy_hand_scale and scale > 0
    count = args.seconds * fps
    assert 0 <= args.first_frame < args.first_frame + count <= len(original['frames'])
    ids = list(range(args.first_frame, args.first_frame + count, args.stride))
    offset = original['frames'][ids[0]]['t']
    duration = count / fps
    preview = {k: copy.deepcopy(v) for k, v in original.items()
               if k not in ('frames', 'qa', 'qa_parts', 'shots')}
    preview.update(fps=fps // args.stride, start=original['start'] + offset,
                   hand_scale=scale, frames=[])
    for n, i in enumerate(ids):
        frame = copy.deepcopy(original['frames'][i])
        assert abs(frame['t'] - offset - n / preview['fps']) < 1e-4
        frame['t'] = n / preview['fps']
        preview['frames'].append(frame)
        assert {k: v for k, v in frame.items() if k != 't'} == {
            k: v for k, v in original['frames'][i].items() if k != 't'}
    for name in ('beats', 'downbeats'):
        preview[name] = [b - offset for b in original.get(name, [])
                         if offset <= b < offset + duration]
    preview['preview_provenance'] = dict(
        scope='Older complete v19 motion, rendered with the current production skin renderer; diagnostic only',
        source=str(args.source.resolve()), source_sha256=source_sha,
        source_frame_indices=ids, source_fps=fps, source_hand_scale=scale,
        original_pose_key_body_data_preserved=True, temporal_interpolation=False,
        newest_static_contact_fixes_integrated=False, full_performance_accepted=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(preview) + '\n')
    assert sha(args.source) == source_sha
    proof = dict(schema='timed-performance-preview/v1', status='prepared',
        source=str(args.source.resolve()), source_sha256=source_sha,
        animation=str(args.output.resolve()), animation_sha256=sha(args.output),
        source_first_frame=ids[0], source_last_frame=ids[-1], source_frame_indices=ids,
        source_start_seconds=preview['start'], duration_seconds=duration,
        source_fps=fps, render_fps=preview['fps'], render_frames=len(ids), hand_scale=scale,
        size_identity_basis='Original v19 run_parts.sh explicitly sets HAND_SCALE=1; no resizing',
        geometry=original.get('geometry', {'key_shape': 'rectangular', 'scope': 'legacy renderer default'}),
        original_pose_key_body_data_preserved=True, source_unchanged_during_run=True,
        newest_static_contact_fixes_integrated=False, full_performance_accepted=False)
    args.output.with_suffix('.preview.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps({k: v for k, v in proof.items() if k != 'source_frame_indices'}, indent=2))


if __name__ == '__main__':
    main()
