# Physical wrist orientation, 2026-10-11

Full objective remains complete natural piano/drummer motion, original music,
controlled digits, physical key relationships and no clipping through any
instrument/body/costume/self geometry throughout continuous time.

Evidence motivating this change: local wrist offsets are copied into solver
frames whose reference rotations differ. Whittaker filtering smooths those
offsets independently and then applies them to different reference matrices.
The forearm penalty uses asin(dot(up/ across, forearm)), which aliases a
backward-facing hand to a straight wrist. This is a code-level observation,
not yet causal attribution for any observed rendered fold.

Introduce two experimental factors, both off by default: absolute physical
history plus SO(3) smoothing; forward/backward-aware forearm direction angles.
History records the actual solved rotation once per physical frame/pass,
transports it to the current reference, and does not transport same-frame
reconciliation scratch states. Joint coordinates/translation are unchanged.
The manifold fit anchors actual rotations and penalizes differences of
spatial angular increments in a common world frame. It is a soft engineering
objective; finite optimizer convergence does not enforce motion/geometry.
Signed flexion uses atan2(up, forward); deviation uses
atan2(across, hypot(up, forward)). Existing bands and weights are unchanged;
these direction coordinates are not a clinical wrist-joint model.

Preregistered matched 2x2: legacy, SO(3), forearm, combined. All four use the
same performed Knights Suite 282.9–286.9 s, full-score compilation, 48 actual
prelude frames, 24 Hz, size 1, seed 0, original score/touch/lift, original
E36 maximum per-frame budgets, notched geometry, no-crossing ownership/root
constraints. Free-control/wrist-carry remain off to isolate these factors.
SO(3) adds a disclosed offline fit (maximum 30 evaluations per hand/pass);
no equal-total-computation claim. All rejected numerical outputs preserved.

Measure all 144 internal and 96 visible poses: original note-onsets/held
coverage, joint ranges/speed, actual wrist angular/linear speed, absolute
rotation orthogonality, hand-forward/forearm angle without asin aliasing,
arm lengths, root/MCP order, idle-middle posture and pad spacing. At identical
preselected visible frames 47/48, independently extract production Blender
level-2 LBS surfaces, verify parity, both quad diagonals, instrument/self/
interhand crossings and finite assigned distal support. Report convergence
and old-baseline reproducibility. Sampled comparisons may support this local
kinematic change only; no continuous clearance, force, full music, naturalness
or complete-performance acceptance. Do not promote defaults when stronger
playing/skin regressions remain. Original pitch20 remains out of keyboard.

Primary reading: Lynch/Park's Northwestern Modern Robotics supplement,
https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-2-3-exponential-coordinates-of-rotation-part-2-of-2/
explains matrix exp/log on SO(3) and integrating angular velocity. The exact
roughness objective above is our engineering design, not a cited paper's
algorithm. IPC feasible-init/positive-distance/filtered-CCD requirements and
Kavan skinning limitations remain in the existing verified source packet.
Use the native five-role graph for methodological criticism, not rig approval.
