# IPC primary-source reading notes

Source: Li et al., *Incremental Potential Contact: Intersection- and
Inversion-free, Large-Deformation Dynamics*, ACM TOG 39(4), 2020.
Primary author-hosted PDF: https://ipc-sim.github.io/file/IPC-paper-350ppi.pdf
Project: https://ipc-sim.github.io/
Read through the web tool on 2026-10-11. The full-resolution PDF was too large;
the 350-ppi primary PDF was readable. These are paraphrases, not direct quotes.

Page 2, introduction: the method combines a smoothed contact barrier with a
custom Newton solver; filtered continuous collision detection restricts line
search so numerical iterates remain feasible.

Page 3, section 2: admissibility is defined for trajectories beginning without
intersection, using continuously positive primitive distances. Nonincident
point/triangle and nonadjacent edge/edge pairs suffice for their mesh model.
The timestep formulation treats the path between nodal configurations as
linear; the paper's guarantees therefore do not automatically cover our
nonlinear articulated FK and skin interpolation.

Page 4, section 3.1: locally defined contact proxies and normal/gap constraints
can vary with rotation, shear and meshing. Their iterative use does not by
itself establish robust behavior across scenes.

Application inference: restoring an already intersecting rig to a validated
feasible pose is a prerequisite, not an IPC guarantee. SLSQP success and static
SAT clearance are insufficient evidence for collision-free animation. Our
actual anatomical interpolation needs its own path representation and
continuous-time collision verification.
