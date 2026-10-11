The drummer controller is a new CPU blockout foundation. The checked-out band
repository contained the piano rig and a drum-blockout TODO, so this does not
retarget an existing drummer animation. Its fingers are retained capsule-chain
grip proxies, not MPFB meshes. Its rebound is prescribed kinematics with a
configurable coefficient; it is not a force or rigid-body simulation.

Run from the repository root with the blockout environment:

```sh
python -m blockout.drums --demo --out /tmp/drums.json
python -m blockout.drum_preview /tmp/drums.json --out /tmp/drums.png
python -m blockout.drum_preview /tmp/drums.json --out /tmp/drums.mp4
python -m unittest discover -s tests -p test_drums.py -v
```

Supply musical hits with `--events hits.json` instead of `--demo`. Input is a list
of `{ "time": 0.5, "drum": "snare", "hand": "R", "velocity": 0.8 }` objects.
Times are seconds; velocities are normalized from 0 to 1. Hand is L, R, or auto.
An object containing `events`, optional `kit`, and optional `rig` can override
geometry and bounds. All distances are metres and angles are radians except the
explicit elbow degree fields. The coordinate frame matches piano: x right,
y away from the performer, z up. Kick and hi-hat foot control are not implemented;
unsupported instruments and impossible same-hand timing raise explicit errors.

Each strike lands the tip's surface on a finite, optionally tilted playing plane.
The stick stays rigid and its grip position is invariant along the shaft. The
elbow stays inside its configured range without changing upper-arm or forearm
length. Reach clipping moves the whole stick/hand together, and a missed target
fails QA. Fingers circumscribe the shaft with separated planes and retained
contact, avoiding the common polygon-chord penetration bug. Grip pressure rises
at impact and relaxes during rebound, while contact remains retained. Pressure
is exported for future anatomical-hand/controller retargeting; it does not yet
drive muscle forces or surface deformation.

The output contains the kit, rig, assigned events, uniform animation frames and
QA. The command exits 2 when QA fails, while preserving output for inspection.
Audit checks every exact event time in addition to at least 240 samples/second:
contact error, all-drum shaft/tip/finger penetration, two-stick clearance, finger
separation, finger-stick clearance/retention, arm/stick lengths, reach, elbow
bounds and wrist/tip speed. Geometry and speed thresholds are explicit engineering
bounds, not population-derived anatomical limits. The fixed-camera preview uses
ordinary Matplotlib/FFmpeg and performs no H3 generation.

Remaining production work: retarget the grip proxies into the anatomical hand
rig, shoulder and wrist anatomical limits, inter-hand finger and arm/torso collisions, drum rim
and shell geometry, pedals, audio onset extraction, and contact impulse dynamics.
The current drum-surface check samples the shaft and time; it is not a continuous
collision proof. Impossible fast transfers or obstructed kit layouts are flagged
rather than silently retimed, stretched, or allowed to count as successful hits.
