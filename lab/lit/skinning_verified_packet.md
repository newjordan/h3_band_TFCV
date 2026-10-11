# Skinning primary-source notes, read 2026-10-11

Blender manual (search-extracted official page; version-4.0 direct fetch failed):
https://docs.blender.org/manual/en/latest/modeling/modifiers/deform/armature.html

The Preserve Volume setting uses quaternion deformation to reduce collapse
around rotating joints. The manual describes a discontinuity around a
180-degree rotation. Neither description establishes self/contact clearance
for our poses. The locally installed Blender 4.0.2 modifier is the executable
authority for the controlled ablation.

Kavan et al., Geometric Skinning with Approximate Dual Quaternion Blending,
2008, primary author PDF, sections 4.1, 4.2 and 6:
https://users.cs.utah.edu/~ladislav/kavan08geometric/kavan08geometric.pdf

The method blends rigid transforms through dual quaternions. Sign selection
matters, and scale/shear need a separate first phase rather than naive rigid
quaternion conversion. The authors describe possible shortest-path flipping
at large rotations. This is a geometric deformation method; it is not a
contact, tissue-force or continuously collision-free simulation certificate.

Local implication (our hypothesis, not a paper result): hold bone poses,
weights, size, key commands and topology fixed and independently audit actual
Blender surfaces with preserve volume off/on. Include a bind-geometry control,
base cage and level-2 limit surface. Report instrument/contact regressions as
well as self crossing changes. A changed surface invalidates old LBS-specific
parity, anchors and trajectory acceptance, even if one fold disappears.
