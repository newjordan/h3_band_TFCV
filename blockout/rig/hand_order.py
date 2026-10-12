"""Score ownership and wrist ordering for performances that forbid crossing.

The score events are immutable. Register order is a hard planning constraint;
capacity/span costs are diagnostics, not proof that every chord is playable.
Wrist separation is an engineering bound, not a skin-clearance certificate.
"""
import numpy as np


def ordered_notes(notes, preferred, key_x, span, tolerance=.05):
    """Allocate every event once, retaining ownership until its original release.

    Work on the complete performed score before extracting a rendering window.
    A new onset cannot put a left-hand note above a still-held right-hand note.
    Simultaneous repeated pitches have one owner; no note is removed or retimed.
    """
    order = sorted(range(len(notes)), key=lambda i: (notes[i].start, notes[i].pitch, i))
    groups = []
    for i in order:
        if not groups or notes[i].start - notes[groups[-1][0]].start >= tolerance:
            groups.append([i])
        else:
            groups[-1].append(i)
    owners, held, problems = {}, [], []
    centres = {'L': None, 'R': None}
    for group in groups:
        time = notes[group[0]].start
        held = [i for i in held if notes[i].end > time]
        pitches = sorted({notes[i].pitch for i in held + group})
        best = None
        for cut in range(len(pitches) + 1):
            lower = set(pitches[:cut])
            proposal = {i: ('L' if notes[i].pitch in lower else 'R') for i in held + group}
            if any(proposal[i] != owners[i] for i in held):
                continue
            cost = 4. * sum(proposal[i] != preferred[i] for i in group)
            sets = {side: sorted({notes[i].pitch for i in held + group if proposal[i] == side})
                    for side in ('L', 'R')}
            for side, ps in sets.items():
                if not ps:
                    continue
                width = key_x(ps[-1]) - key_x(ps[0])
                cost += 100. * max(0, len(ps) - 5) ** 2
                cost += 100. * (max(0., width - span) / .0235) ** 2
                if centres[side] is not None:
                    cost += .05 * ((np.mean([key_x(p) for p in ps]) - centres[side]) / .1) ** 2
            rank = (float(cost), cut)
            if best is None or rank < best[0]:
                best = (rank, proposal, sets)
        if best is None:
            raise ValueError('No ordered ownership preserves previously held events')
        _, proposal, sets = best
        owners.update({i: proposal[i] for i in group})
        for side, ps in sets.items():
            if not ps:
                continue
            centres[side] = float(np.mean([key_x(p) for p in ps]))
            width = key_x(ps[-1]) - key_x(ps[0])
            if len(ps) > 5 or width > span + 1e-12:
                problems.append(dict(time=float(time), hand=side, pitches=ps,
                                     unique_keys=len(ps), span_m=float(width)))
        held += group
    result = {side: [notes[i] for i in order if owners[i] == side] for side in ('R', 'L')}
    changed = [dict(note=i, start=float(notes[i].start), end=float(notes[i].end),
                    pitch=int(notes[i].pitch), original_hand=preferred[i], hand=owners[i])
               for i in order if owners[i] != preferred[i]]
    return result['R'], result['L'], dict(redistributed=changed, capacity_or_span_conflicts=problems,
        original_events=len(notes), allocated_events=len(owners), order_preserved_through_original_releases=True)


def ordered_wrists(right, left, gap=.06, weights=(1., 1.), shoulders=None, arm_lengths=None):
    """Weighted closest ordered x positions, retaining y/z and arm workspace.

    Inputs with shoulders must already lie in the two-bone workspace. Its
    exact x intervals at those y/z are intersected with the ordering halfspace.
    An infeasible intersection raises instead of stretching either arm.
    """
    points = np.array([right, left], dtype=float, copy=True)
    w = np.asarray(weights, float)
    if points.shape != (2, 3) or w.shape != (2,) or not np.isfinite(points).all() or not np.isfinite(w).all():
        raise ValueError('Finite positions and two finite weights are required')
    if np.any(w <= 0) or not np.isfinite(gap) or gap <= 0:
        raise ValueError('Positive ordering gap and weights are required')
    lo, hi = np.full(2, -np.inf), np.full(2, np.inf)
    if shoulders is not None:
        sh = np.asarray(shoulders, float)
        if sh.shape != (2, 3) or not np.isfinite(sh).all() or arm_lengths is None:
            raise ValueError('Both shoulders and physical arm lengths are required')
        inner = abs(arm_lengths[0] - arm_lengths[1]) + 1e-5
        outer = sum(arm_lengths) - 1e-5
        for i in range(2):
            yz2 = float(np.sum((points[i, 1:] - sh[i, 1:]) ** 2))
            if yz2 > outer ** 2 + 1e-12:
                raise ValueError('Wrist reference outside arm workspace')
            extent = np.sqrt(max(0., outer ** 2 - yz2))
            lo[i], hi[i] = sh[i, 0] - extent, sh[i, 0] + extent
            if yz2 < inner ** 2:
                hole = np.sqrt(inner ** 2 - yz2)
                if points[i, 0] < sh[i, 0]:
                    hi[i] = sh[i, 0] - hole
                else:
                    lo[i] = sh[i, 0] + hole
    x = np.clip(points[:, 0], lo, hi)
    if x[0] - x[1] < gap:
        lower, upper = max(lo[1], lo[0] - gap), min(hi[1], hi[0] - gap)
        if lower > upper + 1e-12:
            raise ValueError('Ordered wrists cannot fit the physical arm workspaces')
        left_x = (w[0] * (x[0] - gap) + w[1] * x[1]) / w.sum()
        x[1] = np.clip(left_x, lower, upper)
        x[0] = x[1] + gap
    points[:, 0] = x
    return points[0], points[1]


def wrist_order_audit(frames, fps, gap=.06):
    values = np.array([f['hands']['R'].get('root', {'pos': f['hands']['R']['wrist']})['pos'][0]
                       - f['hands']['L'].get('root', {'pos': f['hands']['L']['wrist']})['pos'][0]
                       for f in frames], float)
    bad = np.flatnonzero(values < gap - 1e-9)
    crossing = np.flatnonzero(values < -1e-9)
    return dict(required_gap_mm=1000. * gap, minimum_gap_mm=float(1000. * values.min()),
        violating_frames=bad.tolist(), crossing_frames=crossing.tolist(),
        crossing_seconds=len(crossing)/fps,
        passed=not len(bad), scope='sampled wrist-root order; skin, containment and nonlinear CCD separate')
