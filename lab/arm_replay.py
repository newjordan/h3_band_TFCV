"""Replay saved hands with continuous, coupled arm IK; never alter wrist targets."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from blockout import performer
from blockout.rig.skeleton import Skeleton
from blockout.rig_qa import segment_closest


def audit(frames):
    motion, lengths, clearance, torso = [], [], [], []
    for i, frame in enumerate(frames):
        body, hands = frame['body'], frame['hands']
        hitbox = performer.torso_box(body)
        for side in ('L', 'R'):
            shoulder = np.array(body['shoulders'][side])
            elbow, wrist = np.array(hands[side]['elbow']), np.array(hands[side]['wrist'])
            if i:
                motion.append(np.linalg.norm(elbow - frames[i - 1]['hands'][side]['elbow']))
            lengths += [abs(np.linalg.norm(elbow - shoulder) - performer.UPPER_ARM),
                        abs(np.linalg.norm(wrist - elbow) - performer.FOREARM)]
            torso.append(max(0., performer.arm_clearance(shoulder, elbow, wrist, hitbox, side)))
        a, b = segment_closest(hands['L']['elbow'], hands['L']['wrist'],
                               hands['R']['elbow'], hands['R']['wrist'])
        clearance.append(np.linalg.norm(a - b) - 0.064)
    return dict(max_elbow_step_mm=float(max(motion)) * 1000,
                p95_elbow_step_mm=float(np.percentile(motion, 95)) * 1000,
                max_bone_length_error_mm=float(max(lengths)) * 1000,
                min_forearm_skin_gap_mm=float(min(clearance)) * 1000,
                forearm_overlap_frames=int(sum(d < -0.003 for d in clearance)),
                max_torso_penetration_mm=float(max(torso)) * 1000)


def replay(source, start, end, speed=4.):
    anim = copy.deepcopy(source)
    skeletons = {s: Skeleton(s) for s in ('L', 'R')}
    previous, result, original = {}, [], []
    dt = 1. / source['fps']
    for frame in anim['frames']:
        absolute = source['start'] + frame['t']
        if not start - 0.5 <= absolute <= end:
            continue
        body, hands = frame['body'], frame['hands']
        shoulders = {s: np.asarray(body['shoulders'][s]) for s in hands}
        wrists = {s: np.asarray(hands[s]['wrist']) for s in hands}
        forwards = {s: np.asarray(hands[s]['root']['rot']) @ skeletons[s].R0['wrist'].T
                   @ skeletons[s].rest_frame[1] for s in hands}
        hitbox = performer.torso_box(body)
        if not previous:
            previous = {s: np.asarray(hands[s]['elbow']) for s in hands}
        original.append(copy.deepcopy(frame)) if absolute >= start else None
        elbows = {}
        for side in hands:
            elbows[side] = performer.arm_ik(shoulders[side], wrists[side], 1 if side == 'R' else -1,
                                             forwards[side], torso=hitbox, previous=previous[side],
                                             dt=dt, max_angular_speed=speed)
        if performer.arm_pair_clearance(shoulders['L'], elbows['L'], wrists['L'],
                                         (shoulders['R'], elbows['R'], wrists['R'])) > 0:
            for _ in range(2):
                for side in hands:
                    other = 'L' if side == 'R' else 'R'
                    elbows[side] = performer.arm_ik(shoulders[side], wrists[side], 1 if side == 'R' else -1,
                                                     forwards[side], torso=hitbox, previous=previous[side],
                                                     other_arm=(shoulders[other], elbows[other], wrists[other]),
                                                     dt=dt, max_angular_speed=speed)
        for side in hands:
            hands[side]['elbow'] = elbows[side].tolist()
        body['elbows'] = {s: e.tolist() for s, e in elbows.items()}
        previous = elbows
        if absolute >= start:
            frame['t'] = round(absolute - start, 4)
            result.append(frame)
    anim['frames'], anim['start'] = result, start
    metrics = dict(start=start, end=end, frames=len(result), max_angular_speed=speed,
                   baseline=audit(original), continuous=audit(result))
    return anim, metrics


def offline_replay(source, start, end, speed=4.):
    anim = copy.deepcopy(source)
    anim['frames'] = [f for f in anim['frames'] if start - .5 <= source['start'] + f['t'] <= end]
    before = copy.deepcopy([f for f in anim['frames'] if source['start'] + f['t'] >= start])
    track_audit = performer.relax_arm_track(anim['frames'], source['fps'], max_angular_speed=speed)
    anim['frames'] = [f for f in anim['frames'] if source['start'] + f['t'] >= start]
    for frame in anim['frames']:
        frame['t'] = round(source['start'] + frame['t'] - start, 4)
    anim['start'] = start
    return anim, dict(start=start, end=end, frames=len(anim['frames']), max_angular_speed=speed,
                       baseline=audit(before), anticipated=audit(anim['frames']), track=track_audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--start', type=float, default=288.)
    parser.add_argument('--end', type=float, default=292.)
    parser.add_argument('--speed', type=float, default=4.)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    solve = offline_replay if args.offline else replay
    anim, metrics = solve(json.loads(args.source.read_text()), args.start, args.end, args.speed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(anim))
    args.output.with_suffix('.metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
