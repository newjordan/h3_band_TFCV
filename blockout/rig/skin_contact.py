"""Constrained restoration of a skinned pose that already intersects the piano.

This is the feasibility stage before a trajectory/contact barrier method can
start. SLSQP success is insufficient: the complete surface and triangle audit
must pass, together with anatomical, temporal, reach and note-anchor checks.
It is a pose experiment, not continuous collision detection or a performance.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares, minimize

from .hand import rotvec
from .contact_placement import calibrated_target, surface_target, REFERENCE
from ..performer import UPPER_ARM, FOREARM
from ..piano_geometry import triangle_box_witness, triangle_box_margin
from ..skin_contacts import key_support
from ..mesh_self import self_intersections, quad_self_intersections
from ..triangle_contact import triangle_separation


def pose_vector(rig, hand):
    return np.r_[hand['root']['pos'], np.zeros(3),
                 [hand['angles'].get(bone, {}).get(dof, 0.)
                  for chain in rig.chains for bone, dof, *_ in chain.dofs]]


def continuous_elbow(shoulder, wrist, reference):
    """Project the reference bend plane onto the exact two-bone elbow circle.

    Unreachable wrists remain measurable through reach constraints. A solve
    never manufactures a longer forearm to make its skin appear collision free.
    """
    shoulder, wrist, reference = map(lambda v: np.asarray(v, float), (shoulder, wrist, reference))
    delta = wrist - shoulder
    distance = np.linalg.norm(delta)
    u = delta / max(distance, 1e-10)
    if distance >= UPPER_ARM + FOREARM or distance <= abs(UPPER_ARM - FOREARM):
        return shoulder + UPPER_ARM * u
    a = (UPPER_ARM ** 2 - FOREARM ** 2 + distance ** 2) / (2 * distance)
    center = shoulder + a * u
    radial = reference - center
    radial -= u * (radial @ u)
    if np.linalg.norm(radial) < 1e-10:
        radial = np.cross(u, np.eye(3)[np.argmin(np.abs(u))])
    return center + radial / np.linalg.norm(radial) * np.sqrt(max(0., UPPER_ARM ** 2 - a ** 2))


@dataclass
class SkinAnchor:
    finger: int
    pitch: int
    vertex: int
    target: np.ndarray
    box: int
    placement: dict = None
    tangential_tolerance: float = .002


def musical_anchors(skin, operator, vertices, hand, scene, keys, minimum_activation=.7,
                    mode='support', required_pitches=(), placement='source', reference=REFERENCE,
                    active_pitches=()):
    """Use actual distal skin for supported notes already on their assigned key.

    Existing pad locations identify the key and a small distal contact region.
    The contact point is the lowest rendered skin in that region, rather than
    the skin point nearest an approximate pad that may lie inside the mesh.
    Its existing tangential placement is retained on the rotating key surface.
    Missing score contacts remain missing and must be planned separately.
    """
    from ..contact import on_key
    if mode not in ('support', 'nearest_pad'):
        raise ValueError('Unknown contact-anchor selection mode')
    if placement not in ('source', 'calibrated', 'calibrated_soft'):
        raise ValueError('Unknown contact-placement prior')
    active_pitches = set(active_pitches) | set(required_pitches) | {
        int(p) for p, a in zip(hand['assigned_keys'], hand['activation'])
        if p is not None and a >= minimum_activation}
    anchors = []
    for f, (pitch, pad, active) in enumerate(zip(hand['assigned_keys'], hand['pads'], hand['activation']), 1):
        required = pitch is not None and int(pitch) in required_pitches
        if pitch is None or (not required and (active < minimum_activation or not on_key(pitch, pad, keys))):
            continue
        point = np.asarray(pad, float)
        box, _ = surface_target(point, pitch, scene)
        # A distal vertex, not a web/palm point accidentally nearby in world space.
        column = skin.bone_index[f'finger{f}-3']
        influenced = np.flatnonzero(np.asarray(operator @ skin.weights[:, column]) > .2)
        local = (vertices[influenced] - scene.origins[box]) @ scene.rotations[box]
        pad_local = (point - scene.origins[box]) @ scene.rotations[box]
        vertex = influenced[np.argmin(np.linalg.norm(vertices[influenced] - point, axis=1))]
        target_local = pad_local.copy()
        if mode == 'support':
            nearby = np.linalg.norm(local[:, :2] - pad_local[:2], axis=1) <= .008
            footprint = np.all((local[:, :2] >= scene.lower[box, :2] + .001)
                               & (local[:, :2] <= scene.upper[box, :2] - .001), axis=1)
            candidates = np.flatnonzero(nearby & footprint)
            candidates = np.array([i for i in candidates if on_key(pitch, vertices[influenced[i]], keys)], int)
            if required and not len(candidates):
                # A sustained note may have slipped off the front in the source
                # plan. Retarget the underside patch to the key; never choose a
                # dorsal point merely because it is closer to an internal pad.
                candidates = np.flatnonzero(nearby)
            if len(candidates):
                chosen = candidates[np.argmin(local[candidates, 2])]
                vertex, target_local = influenced[chosen], local[chosen].copy()
        metadata = None
        if placement in ('calibrated', 'calibrated_soft'):
            box, target, metadata = calibrated_target(f, pitch, scene, keys, active_pitches, reference)
        else:
            source_point = target_local @ scene.rotations[box].T + scene.origins[box]
            box, target = surface_target(source_point, pitch, scene)
        anchors.append(SkinAnchor(f, int(pitch), int(vertex), target, box, metadata,
                                  None if placement == 'calibrated_soft' else .002))
    return anchors


def restore_pose(rig, skin, surface, hand, scene, shoulder, previous, following, dt,
                 anchors=(), allow_wrist=True, max_joint_speed=12., max_wrist_speed=.8,
                 wrist_range=.035, iterations=140, passes=4, gap=1e-5,
                 previous_hand=None, following_hand=None, max_root_speed=4., max_elbow_speed=4.,
                 triangle_mode='sat', feasibility_iterations=0, human_prior=None, human_weight=0.,
                 anchor_objective=False, initial_seed=None, mesh_self_constraints=True):
    operator, faces = surface
    triangles = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]],
                                faces[:, [1, 2, 3]], faces[:, [1, 3, 0]]])
    initial = pose_vector(rig, hand)
    reference_rot = np.asarray(hand['root']['rot'], float)
    shoulder, reference_elbow = np.asarray(shoulder, float), np.asarray(hand['elbow'], float)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('A physical timestep is required')
    if rig.side != skin.side or not np.isclose(rig.sk.scale, skin.sk.scale):
        raise ValueError('The solver and rendered skin must use the same hand and scale')
    if triangle_mode not in ('sat', 'witness'):
        raise ValueError('Unknown triangle constraint mode')
    if not np.isfinite(human_weight) or human_weight < 0 or (human_weight and human_prior is None):
        raise ValueError('A finite nonnegative human weight requires its reference prior')
    if (not all(np.isfinite(v) and v > 0 for v in
                (max_joint_speed, max_wrist_speed, wrist_range, max_root_speed, max_elbow_speed))
            or not isinstance(iterations, int) or iterations < 1
            or not isinstance(passes, int) or passes < 1
            or not isinstance(feasibility_iterations, int) or feasibility_iterations < 0
            or not np.isfinite(gap) or gap < 0):
        raise ValueError('Motion bounds and solver budgets must be finite and positive')
    for neighbor in (previous, following):
        if neighbor is not None and (np.shape(neighbor) != initial.shape or not np.isfinite(neighbor).all()):
            raise ValueError('Neighbors must be finite physical poses of the same rig')
    joint_lo, joint_hi = rig.lo.copy(), rig.hi.copy()
    for neighbor in (previous, following):
        if neighbor is not None:
            joint_lo = np.maximum(joint_lo, neighbor[6:] - max_joint_speed * dt)
            joint_hi = np.minimum(joint_hi, neighbor[6:] + max_joint_speed * dt)
    if np.any(joint_lo > joint_hi):
        return dict(accepted=False, reason='Neighbor joint speed bounds have no intersection'), hand
    # Translation variables in mm avoid severely ill-conditioned numerical
    # gradients alongside joint angles in radians. Finger anatomy stays fixed.
    bounds = [(-wrist_range * 1000, wrist_range * 1000)] * 3 + [(-.3, .3)] * 3
    if not allow_wrist:
        bounds = [(0., 0.)] * 6
    bounds += list(zip(joint_lo, joint_hi))
    z = np.r_[np.zeros(6), np.clip(initial[6:], joint_lo, joint_hi)]
    if initial_seed is not None:
        initial_seed = np.asarray(initial_seed, float)
        if initial_seed.shape != z.shape or not np.isfinite(initial_seed).all():
            raise ValueError('Initialization must be a finite vector in solver coordinates')
        seed_lower, seed_upper = np.array(bounds).T
        z = np.clip(initial_seed, seed_lower, seed_upper)
    targets = np.array(hand['pads'])
    activity = np.clip(np.array(hand['activation']), .05, 1.)
    anchor_operator = operator[[anchor.vertex for anchor in anchors]]
    anchored_fingers = {anchor.finger - 1 for anchor in anchors} if anchor_objective else set()
    cache = {}

    def evaluate(value):
        if 'z' in cache and np.array_equal(value, cache['z']):
            return cache['pose']
        x = np.r_[initial[:3] + value[:3] / 1000., value[3:]]
        root_rot, root_pos, points = rig.points(x, reference_rot)
        pads = np.asarray(rig.effectors(points, rig._rots))
        elbow = continuous_elbow(shoulder, root_pos, reference_elbow)
        base = skin.pose(root_rot, root_pos, rig.angles(x), elbow)
        result = (x, root_rot, root_pos, pads, elbow, base)
        landmarks = np.asarray([root_pos] + [p for chain in points for p in chain], float)
        cache.update(z=value.copy(), pose=result, landmarks=landmarks)
        return result

    def objective(value):
        x, _, _, pads, _, base = evaluate(value)
        pad_error = (pads - targets) * 1000. / 4.
        if anchored_fingers:
            pad_error[list(anchored_fingers)] = 0.
        contact_cost = 0.
        if anchor_objective and anchors:
            for anchor, point in zip(anchors, anchor_operator @ base):
                if anchor.tangential_tolerance is not None:
                    contact_cost += float(np.sum(((point - anchor.target) * 1000. / 4.) ** 2))
                else:
                    local = (point - scene.origins[anchor.box]) @ scene.rotations[anchor.box]
                    target = (anchor.target - scene.origins[anchor.box]) @ scene.rotations[anchor.box]
                    lower, upper = np.asarray(anchor.placement['depth_iqr_mm']) / 1000.
                    sigma = max(.008, (upper - lower) / 1.349)
                    excess = float(point[1] - np.clip(point[1], lower, upper)) / sigma
                    contact_cost += ((local[0] - target[0]) / .004) ** 2 + ((local[2] - target[2]) / .004) ** 2
                    contact_cost += 2. * (np.sqrt(1. + excess ** 2) - 1.)
                    contact_cost += .05 * ((point[1] - anchor.target[1]) / sigma) ** 2
        form_cost = human_prior.evaluate(cache['landmarks'], rig.side, hand['activation'])['cost'] if human_weight else 0.
        return (float(np.sum(pad_error ** 2 * activity[:, None] ** 2)) + contact_cost + human_weight * form_cost
                + float(np.sum((value[:3] / 12.) ** 2))
                + float(np.sum((value[3:6] / .2) ** 2))
                + .15 * float(np.sum(((x[6:] - initial[6:]) / .3) ** 2)))

    all_initial = operator @ evaluate(z)[-1]
    initial_form = human_prior.evaluate(cache['landmarks'], rig.side, hand['activation']) if human_prior is not None else None
    initial_distance, _ = scene.closest(all_initial)
    boxes = scene.candidates(all_initial, padding=wrist_range + .02)
    active = set(np.flatnonzero(initial_distance < .002).tolist())
    active.update(anchor.vertex for anchor in anchors)
    witnesses, history, triangle_pairs = [], [], set()
    skin_triangles = np.concatenate([faces[:, [0, 1, 2]], faces[:, [0, 2, 3]]])
    skin_pairs = set()
    if mesh_self_constraints:
        initial_mesh = self_intersections(all_initial, skin_triangles)
        skin_pairs.update(tuple(p) for p in initial_mesh['pairs']
                          if not set(skin_triangles[p[0]]) & set(skin_triangles[p[1]]))
    last_result = None
    for outer in range(passes):
        ids = np.array(sorted(active), int)
        selected = operator[ids]
        if witnesses:
            selected = sparse.vstack([selected] + witnesses, format='csr')
        active_triangles = {}
        for box, ti in sorted(triangle_pairs):
            active_triangles.setdefault(box, []).append(ti)
        triangle_stencils = {box: operator[triangles[ids].ravel()]
                             for box, ids in active_triangles.items()}
        skin_pair_indices = np.asarray(sorted(skin_pairs), int).reshape(-1, 2)
        skin_stencil = operator[skin_triangles[skin_pair_indices].ravel()] if len(skin_pair_indices) else None

        def constraints(value):
            x, root_rotation, root, _, current_elbow, base = evaluate(value)
            points = selected @ base
            distance, _ = scene.closest(points, boxes)
            result = [(distance - gap) * 1000.]
            for box, stencil in triangle_stencils.items():
                triangle = (stencil @ base).reshape(-1, 3, 3)
                local = (triangle - scene.origins[box]) @ scene.rotations[box]
                result.append((triangle_box_margin(local, scene.lower[box], scene.upper[box]) - gap) * 1000.)
            if skin_stencil is not None:
                paired = (skin_stencil @ base).reshape(-1, 2, 3, 3)
                result.append((triangle_separation(paired[:, 0], paired[:, 1]) - gap) * 1000.)
            reach = np.linalg.norm(root - shoulder)
            result.append(np.array([reach - abs(UPPER_ARM - FOREARM) - 1e-5,
                                    UPPER_ARM + FOREARM - 1e-5 - reach]) * 1000.)
            for neighbor in (previous, following):
                if neighbor is not None:
                    result.append([1000. * (max_wrist_speed * dt - np.linalg.norm(root - neighbor[:3]))])
            if previous_hand is not None or following_hand is not None:
                # The full neighboring poses retain orientation/elbow evidence
                # omitted by the historical joint-only vector representation.
                from .motion import rotation_angle, bend_plane_angle
                for neighbor in (previous_hand, following_hand):
                    if neighbor is not None:
                        # Convert angular slacks to a comparable numerical
                        # length scale (50 mm/rad). This changes conditioning,
                        # never the zero crossing of either hard speed bound.
                        result.append([50. * (max_root_speed * dt - rotation_angle(root_rotation, neighbor['root']['rot'])),
                                       50. * (max_elbow_speed * dt - bend_plane_angle(shoulder, root, current_elbow, neighbor['elbow']))])
            if anchors:
                contacts = anchor_operator @ base
                for anchor, point in zip(anchors, contacts):
                    local = (point - scene.origins[anchor.box]) @ scene.rotations[anchor.box]
                    target_local = (anchor.target - scene.origins[anchor.box]) @ scene.rotations[anchor.box]
                    # Keep placement within 2 mm and the actual skin supported
                    # at the solid top, outside by at most 0.1 mm.
                    if anchor.tangential_tolerance is not None:
                        result.append((anchor.tangential_tolerance - np.abs(local[:2] - target_local[:2])) * 1000.)
                    # Tangential placement tolerance is not permission to move
                    # off the finite top. Keep the actual skin point inside it.
                    result.append((local[:2] - scene.lower[anchor.box, :2] - gap) * 1000.)
                    result.append((scene.upper[anchor.box, :2] - local[:2] - gap) * 1000.)
                    top_gap = local[2] - scene.upper[anchor.box, 2]
                    # The contact must sit above the assigned top. A signed
                    # distance from the union of solids alone would also allow
                    # an anchor below the underside of a key.
                    result.append([1000. * (top_gap - gap), 1000. * (.0001 - top_gap)])
            result.append(rig.finger_clearances(x, reference_rot) * 1000.)
            return np.concatenate([np.ravel(r) for r in result])

        seed_result = None
        seed_minimum = float(constraints(z).min())
        if feasibility_iterations and seed_minimum < -1e-3:
            # A constrained descent from an intersecting source can make a
            # large, infeasible first step. First reduce violated constraints
            # inside the identical hard bounds with a trust-region solve.
            # This is initialization, never an alternative acceptance gate.
            lower, upper = np.array(bounds).T
            free = np.flatnonzero(upper > lower)
            seed_base = np.clip(z, lower, upper)
            def expand(value):
                candidate = seed_base.copy()
                candidate[free] = value
                return candidate
            if len(free):
                seed_result = least_squares(lambda value: np.minimum(constraints(expand(value)), 0.),
                    seed_base[free], bounds=(lower[free], upper[free]),
                    x_scale=np.r_[np.full(3, 12.), np.full(3, .2), np.full(len(initial) - 6, .3)][free],
                    max_nfev=feasibility_iterations, ftol=1e-9, xtol=1e-9, gtol=1e-9)
                seed = expand(np.clip(seed_result.x, lower[free], upper[free]))
                if np.isfinite(seed).all() and constraints(seed).min() > seed_minimum:
                    z = seed
        seed = z.copy()
        seed_minimum = float(constraints(seed).min())
        last_result = minimize(objective, z, method='SLSQP', bounds=bounds,
                               constraints=[dict(type='ineq', fun=constraints)],
                               options=dict(maxiter=iterations, ftol=1e-7, eps=1e-6))
        z = last_result.x
        candidate_source = 'SLSQP'
        lower, upper = np.array(bounds).T
        if feasibility_iterations and (not np.isfinite(z).all() or np.any(z < lower - 1e-9)
                                      or np.any(z > upper + 1e-9)
                                      or constraints(z).min() < min(seed_minimum, -1e-3) - 1e-6):
            # Retain the better bounded initialization when the next numerical
            # step loses feasibility. The full surface gate still runs below.
            z = seed
            candidate_source = 'bounded feasibility seed'
        base = evaluate(z)[-1]
        vertices = operator @ base
        mesh_result = self_intersections(vertices, skin_triangles) if mesh_self_constraints else None
        if mesh_result is not None:
            skin_pairs.update(tuple(p) for p in mesh_result['pairs']
                              if not set(skin_triangles[p[0]]) & set(skin_triangles[p[1]]))
        distances, _ = scene.closest(vertices, boxes)
        contacts = scene.triangle_contacts(vertices, triangles)
        active.update(np.flatnonzero(distances < .002).tolist())
        # Add real clipped-triangle witnesses. A vertex-only active set can
        # otherwise falsely report feasibility at a solid edge or key gap.
        added = 0
        for box, triangle_ids in contacts:
            for ti in triangle_ids:
                if triangle_mode == 'sat':
                    triangle_pairs.add((box, int(ti)))
                    added += 1
                    continue
                triangle = triangles[ti]
                local = (vertices[triangle] - scene.origins[box]) @ scene.rotations[box]
                barycentric = triangle_box_witness(local, scene.lower[box], scene.upper[box])
                if barycentric is not None:
                    witnesses.append(sparse.csr_matrix(barycentric[None]) @ operator[triangle])
                    added += 1
        history.append(dict(pass_index=outer, optimizer_success=bool(last_result.success),
                            message=str(last_result.message), iterations=int(last_result.nit),
                            active_vertices=len(ids), triangle_witnesses_added=added,
                            triangle_constraint_mode=triangle_mode, active_triangle_constraints=len(triangle_pairs),
                            feasibility_seed_evaluations=int(seed_result.nfev) if seed_result is not None else 0,
                            feasibility_seed_constraint_minimum=seed_minimum,
                            candidate_source=candidate_source,
                            minimum_clearance_mm=float(distances.min() * 1000.),
                            constraint_minimum_mm=float(constraints(z).min()),
                            triangle_intersections=sum(len(t) for _, t in contacts)))
        history[-1]['skin_self_intersections'] = mesh_result['unexpected_surface_intersections'] if mesh_result else None
        history[-1]['active_skin_pair_constraints'] = len(skin_pairs)
        if (distances.min() >= -1e-5 and not contacts and constraints(z).min() >= -1e-3
                and (mesh_result is None or mesh_result['surface_intersection_clear'])):
            break
    x, root_rot, root_pos, pads, elbow, base = evaluate(z)
    all_vertices = operator @ base
    distances, _ = scene.closest(all_vertices)
    contacts = scene.triangle_contacts(all_vertices, triangles)
    import copy
    corrected = copy.deepcopy(hand)
    corrected.update(wrist=root_pos.tolist(), angles=rig.angles(x), elbow=elbow.tolist(), pads=pads.tolist())
    corrected['root'].update(pos=root_pos.tolist(), rot=root_rot.tolist())
    _, _, points = rig.points(x, reference_rot)
    corrected['fingers'] = [np.array(p).tolist() for p in points]
    speed = max((float(np.max(np.abs(x[6:] - n[6:])) / dt)
                 for n in (previous, following) if n is not None), default=0.)
    anchor_result = []
    for anchor, point in zip(anchors, anchor_operator @ base):
        local = (point - scene.origins[anchor.box]) @ scene.rotations[anchor.box]
        target = (anchor.target - scene.origins[anchor.box]) @ scene.rotations[anchor.box]
        anchor_result.append(dict(finger=anchor.finger, pitch=anchor.pitch,
            top_gap_mm=float((local[2] - scene.upper[anchor.box, 2]) * 1000.),
            placement_error_mm=float(np.max(np.abs(local[:2] - target[:2])) * 1000.),
            contact_depth_mm=float(point[1] * 1000.), requested_depth_mm=float(anchor.target[1] * 1000.),
            placement_prior=anchor.placement,
            tangential_placement_hard=anchor.tangential_tolerance is not None,
            exposed_top_support=bool(key_support(all_vertices, [anchor.vertex], anchor.pitch, scene)['supported'])))
    reach_error = max(abs(np.linalg.norm(elbow - shoulder) - UPPER_ARM),
                      abs(np.linalg.norm(root_pos - elbow) - FOREARM))
    lower, upper = np.array(bounds).T
    bounds_violation = float(max(0., (lower - z).max(), (z - upper).max()))
    accepted = (distances.min() >= -1e-5 and not contacts and constraints(z).min() >= -1e-3
                and speed <= max_joint_speed + 1e-6 and reach_error < 1e-6 and bounds_violation < 1e-9
                and all(a['exposed_top_support'] for a in anchor_result))
    report = dict(accepted=bool(accepted), scope='one hand pose; instrument skin and same-hand surface intersections, proxy finger spacing, two-sided joint/wrist speed when neighbors supplied, fixed shoulders; no interhand/body/continuous-time proof',
                  trajectory_validated=False, optimizer_success=bool(last_result.success),
                  initial_penetration_mm=float(max(0., -initial_distance.min()) * 1000.),
                  final_penetration_mm=float(max(0., -distances.min()) * 1000.),
                  triangle_intersections=sum(len(t) for _, t in contacts),
                  wrist_displacement_mm=((root_pos - initial[:3]) * 1000.).tolist(),
                  maximum_joint_speed_rad_s=speed, arm_length_error_mm=reach_error * 1000.,
                  configuration_bounds_violation=bounds_violation,
                  feasibility_iterations=feasibility_iterations,
                  initialization='bounded supplied seed' if initial_seed is not None else 'bounded source pose',
                  proxy_self_clearance_mm=float(rig.finger_clearances(x, reference_rot).min() * 1000.),
                  anchors=anchor_result, history=history)
    final_self = quad_self_intersections(all_vertices, faces)
    # Acceptance always requires the full self surface gate. Disabling active
    # constraints is an ablation of optimization, not permission to skip QA.
    report['mesh_self'] = final_self
    report['mesh_self_constraints_enabled'] = mesh_self_constraints
    accepted = bool(accepted and final_self['surface_intersection_clear'])
    report['accepted'] = accepted
    report['contact_objective'] = 'actual anchor skin for held fingers; source pads for free fingers' if anchor_objective else 'source approximate pads'
    if human_prior is not None:
        evaluate(z)
        report['human_form'] = dict(weight=human_weight, before=initial_form,
            numerical_iterate=human_prior.evaluate(cache['landmarks'], rig.side, hand['activation']),
            acceptance_scope='soft guidance only; no human naturalness pass')
    report['orientation_temporal_bounds_available'] = previous_hand is not None or following_hand is not None
    if report['orientation_temporal_bounds_available']:
        from .motion import rotation_angle, bend_plane_angle
        neighbors = [n for n in (previous_hand, following_hand) if n is not None]
        report['maximum_wrist_angular_speed_rad_s'] = max(rotation_angle(root_rot, n['root']['rot']) / dt for n in neighbors)
        report['maximum_elbow_plane_speed_rad_s'] = max(bend_plane_angle(shoulder, root_pos, elbow, n['elbow']) / dt for n in neighbors)
        report['accepted'] = bool(report['accepted'] and report['maximum_wrist_angular_speed_rad_s'] <= max_root_speed + 1e-6
                                  and report['maximum_elbow_plane_speed_rad_s'] <= max_elbow_speed + 1e-6)
        accepted = report['accepted']
    # A rejected numerical iterate is evidence, never a replacement hand pose.
    # Preserve the source even if SLSQP diverged far into a solid.
    report['candidate_exported'] = bool(accepted)
    return report, corrected if accepted else copy.deepcopy(hand)
