"""Bone-parented, silhouette-driven MPFB hand looks for the piano normal/Canny plate.

Skeleton replaces the visible skin with waisted phalanges, condyles, metacarpals,
carpals and paired radius/ulna shafts. Monster adds talons, warts and dorsal scales
and spikes. Robot adds bevelled segment shells, hinge pins, panels and hydraulics.
Paw adds separate furry toes, little claws, thin beans, a trefoil pad and a tufted
cuff. Real takes the same entry point and adds nothing.

The rig, skin vertices, poses and wrist scale are never changed. Only skeleton
hides the skin. All dimensions derive from rest-skin vertex-group measurements
(weight > .6); every object is bone-parented with an identity parent inverse and
tail-relative coordinates. Dorsal additions keep the key-contact side slim;
arched claws root behind the tips and pads are inset to target no more than
about 4 mm below the native fingertip pad. Toes remain separate. Robot shells
are 1.20x the measured radius; paw volumes are 1.35-1.40x. Existing gauntlets
are independent. Each style stays below 3,000 total scene objects.
Every addition carries hand_style and the supplied grey skin material, including
in the mask pass. Low-resolution primitive meshes are shared across instances;
there are no per-object modifiers, operators, external packages or frame updates.
"""

import math

import bpy
from mathutils import Matrix, Vector


STYLES = ("skeleton", "monster", "robot", "paw", "real")


def _lathe(name, rings, sides=12, smooth=True):
    """Make a capped Y-axis profile; each ring is (y, radius)."""
    verts = [(r * math.cos(2 * math.pi * j / sides), y,
              r * math.sin(2 * math.pi * j / sides))
             for y, r in rings for j in range(sides)]
    faces = [tuple(reversed(range(sides))),
             tuple((len(rings) - 1) * sides + j for j in range(sides))]
    for k in range(len(rings) - 1):
        for j in range(sides):
            a, b = k * sides + j, k * sides + (j + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    return _mesh(name, verts, faces, smooth)


def _mesh(name, verts, faces, smooth):
    me = bpy.data.meshes.new("hand_style_" + name)
    # Profiles turn from +X towards +Z around +Y; reverse for outward normals.
    me.from_pydata(verts, [], [tuple(reversed(face)) for face in faces])
    me.update()
    for p in me.polygons:
        p.use_smooth = smooth
    return me


def _primitives():
    # Sphere from a lathe: 12 meridians x 11 rings, shared even by tiny rivets.
    ball = _lathe("ball", [(math.cos(math.pi * i / 10),
                              max(.001, math.sin(math.pi * i / 10)))
                             for i in range(10, -1, -1)])
    shaft = _lathe("bone", [(-.5, .78), (-.43, 1.12), (-.30, .78),
                            (0, .65), (.30, .78), (.43, 1.12), (.5, .78)])
    tube = _lathe("cylinder", [(-.5, 1), (.5, 1)], 12, False)
    # Bevelled box, including bevelled end edges, without a modifier per object.
    octagon = [(-.70, -1), (.70, -1), (1, -.70), (1, .70),
               (.70, 1), (-.70, 1), (-1, .70), (-1, -.70)]
    verts = [(x * r, y, z * r) for y, r in
             ((-.5, .82), (-.43, 1), (.43, 1), (.5, .82)) for x, z in octagon]
    faces = [tuple(reversed(range(8))), tuple(range(24, 32))]
    faces += [(k * 8 + j, k * 8 + (j + 1) % 8,
               (k + 1) * 8 + (j + 1) % 8, (k + 1) * 8 + j)
              for k in range(3) for j in range(8)]
    shell = _mesh("bevel_box", verts, faces, False)
    tuft = _lathe("soft_tuft", [(0, .72), (.12, 1), (.40, .72),
                                (.72, .36), (.93, .12), (1, .015)], 8)
    spike = _lathe("spike", [(0, 1), (.15, .88), (1, .015)], 8)
    # Curved hook, starting on the nail side and ending forward and slightly down.
    centres = [(0, 0, 1), (.25, .30, .86), (.55, .32, .60),
               (.82, .12, .30), (1, -.12, .008)]
    verts = [(r * math.cos(2 * math.pi * j / 10), y,
              z + .16 * r * math.sin(2 * math.pi * j / 10))
             for y, z, r in centres for j in range(10)]
    faces = [tuple(reversed(range(10)))]
    faces += [(k * 10 + j, k * 10 + (j + 1) % 10,
               (k + 1) * 10 + (j + 1) % 10, (k + 1) * 10 + j)
              for k in range(4) for j in range(10)]
    faces.append(tuple(range(40, 50)))
    hook = _mesh("hook", verts, faces, True)
    return dict(ball=ball, shaft=shaft, tube=tube, shell=shell,
                tuft=tuft, spike=spike, hook=hook)


def build(style, arm, mesh, materials, collection):
    """Build once at rest; materials is {'skin': M_SKIN}. Return added objects.

    Canonical coordinates for each bone are X across, Y head-to-tail, Z dorsal,
    with Y=0 at its tail. A per-bone basis maps these into Blender's bone frame.
    Palm panels use a separate rigid wrist-parented anatomical palm frame.
    """
    if style not in STYLES:
        raise ValueError("Unknown hand style %r; choose %s" % (style, ", ".join(STYLES)))
    if style == "real":
        return []
    if style == "skeleton":
        mesh.hide_render = True

    groups = {g.index: g.name for g in mesh.vertex_groups}
    points = {}
    world_to_arm = arm.matrix_world.inverted()
    for v in mesh.data.vertices:
        p = world_to_arm @ mesh.matrix_world @ v.co
        for g in v.groups:
            if g.weight > .6:
                points.setdefault(groups[g.group], []).append(p)

    # Read-only rest-surface queries seat scales and pads in the actual skin,
    # whose cross section is not necessarily centred on the bone axis.
    bpy.context.view_layer.update()
    rest_skin = mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
    arm_to_skin = mesh.matrix_world.inverted() @ arm.matrix_world
    skin_to_arm = arm_to_skin.inverted()

    primitives = _primitives()
    for me in primitives.values():
        me.materials.append(materials["skin"])
    made = []

    for side in "LR":
        bones = {b.name[:-2]: b for b in arm.data.bones if b.name.endswith("." + side)}
        wrist = bones["wrist"]
        mcps = [bones[f"finger{f}-1"].head_local for f in range(2, 6)]
        centre = sum(mcps, Vector()) / 4
        forward = (centre - wrist.head_local).normalized()
        across = mcps[-1] - mcps[0]
        across = (across - forward * across.dot(forward)).normalized()
        dorsal = across.cross(forward) * (1 if side == "R" else -1)
        # Right-handed frame on both sides; the palm itself is mirrored by its data.
        across = forward.cross(dorsal).normalized()
        palm_basis = Matrix((across, forward, dorsal)).transposed()
        radii, bases = {}, {}
        for name, b in bones.items():
            along = (b.tail_local - b.head_local).normalized()
            rr = sorted(((p - b.head_local) - along * (p - b.head_local).dot(along)).length
                        for p in points.get(b.name, []))
            # Sparse vertex groups fall back to the same side's measured wrist.
            radii[name] = rr[len(rr) // 2] if rr else radii.get("wrist", wrist.length * .65)
            z = (dorsal - along * dorsal.dot(along)).normalized()
            basis = Matrix((along.cross(z), along, z)).transposed()
            bases[name] = b.matrix_local.to_3x3().inverted() @ basis

        def put(kind, label, bone, pos, size, rotation=None, palm=False):
            b = bones[bone]
            ob = bpy.data.objects.new(f"{style}_{side}_{label}", primitives[kind])
            collection.objects.link(ob)
            ob["hand_style"] = style
            ob.parent, ob.parent_type, ob.parent_bone = arm, "BONE", b.name
            ob.matrix_parent_inverse = Matrix.Identity(4)
            if palm:
                basis = b.matrix_local.to_3x3().inverted() @ palm_basis
                location = b.matrix_local.inverted() @ (palm_origin + palm_basis @ Vector(pos))
                location.y -= b.length
            else:
                basis = bases[bone]
                location = basis @ Vector(pos)
            ob.location = location
            ob.rotation_euler = (basis @ (rotation or Matrix.Identity(3))).to_euler()
            ob.scale = size
            made.append(ob)
            return ob

        def line(kind, label, bone, a, b, radius, palm=False):
            a, b = Vector(a), Vector(b)
            d = b - a
            rot = Vector((0, 1, 0)).rotation_difference(d.normalized()).to_matrix()
            origin = a if kind in ("tuft", "spike") else (a + b) * .5
            return put(kind, label, bone, origin, (radius, d.length, radius), rot, palm)

        palm_origin = wrist.head_local.copy()
        # Surface extents measured in a palm-aligned frame; no hard-coded hand size.
        hand_points = [p for name, ps in points.items() if name.endswith("." + side)
                       and (name.startswith("metacarpal") or name.startswith("wrist")) for p in ps]
        palm_points = [palm_basis.transposed() @ (p - palm_origin) for p in hand_points]
        mcp_local = [palm_basis.transposed() @ (p - palm_origin) for p in mcps]
        px = sum(p.x for p in mcp_local) / 4
        palm_length = (centre - palm_origin).length
        palm_width = (max(p.x for p in mcp_local) - min(p.x for p in mcp_local)) * .5
        top = sorted(p.z for p in palm_points)[int(.80 * (len(palm_points) - 1))]
        wr = radii["wrist"]

        def surface(bone, x, y, back=True, palm=False):
            b, r = bones[bone], radii[bone]
            basis = palm_basis if palm else b.matrix_local.to_3x3() @ bases[bone]
            origin = (palm_origin if palm else b.tail_local) + basis @ Vector((x, y, 0))
            up = basis.col[2]
            sign = 1 if back else -1
            start = arm_to_skin @ (origin + up * (8 * r * sign))
            direction = (arm_to_skin.to_3x3() @ (-up * sign)).normalized()
            hit, co, _, _ = rest_skin.ray_cast(start, direction, distance=16 * r)
            return (skin_to_arm @ co - origin).dot(up) if hit else (.7 * r * sign)

        if style == "skeleton":
            for name in [f"finger{f}-{k}" for f in range(1, 6) for k in (1, 2, 3)] + [
                    f"metacarpal{m}" for m in range(1, 5)]:
                b, r = bones[name], radii[name]
                sr = r * (.24 if name.startswith("metacarpal") else .40)
                put("shaft", name + "_shaft", name, (0, -.5 * b.length, 0),
                    (sr, .88 * b.length, sr * .84))
                # Two rounded condyles make a recognizable knuckle, not a smooth tube.
                for x in (-.52, .52):
                    put("ball", name + "_condyle", name, (x * sr, -b.length + sr * .65, 0),
                        (sr * .88, sr * .95, sr * .85))
                if name.endswith("-3"):
                    put("ball", name + "_tuft", name, (0, -sr * .35, 0),
                        (sr * .95, sr * 1.30, sr * .85))
            # Eight carpal pebbles, near the wrist head and the metacarpal roots.
            for row in range(2):
                for j in range(4):
                    put("ball", "carpal", "wrist",
                        (px + (j - 1.5) * wr * .43, palm_length * (.10 + row * .16), 0),
                        (wr * .24, wr * .27, wr * .22), palm=True)

        for f in range(1, 6):
            for k in (1, 2, 3):
                name = f"finger{f}-{k}"
                b, r = bones[name], radii[name]
                length = b.length
                if style == "monster":
                    for t in (.24, .53, .80):
                        z = surface(name, .22 * r, -length * t)
                        put("ball", name + "_wart", name, (.22 * r, -length * t, z + .04 * r),
                            (.46 * r, .42 * r, .36 * r))
                    z = surface(name, -.20 * r, -.57 * length)
                    line("spike", name + "_ridge", name,
                         (-.20 * r, -.57 * length, z - .06 * r),
                         (-.20 * r, -.47 * length, z + .65 * r), .32 * r)
                elif style == "robot":
                    # Width 1.20r, dorsal shell; the palm side stays within the skin pad.
                    put("shell", name + "_shell", name, (0, -.5 * length, .23 * r),
                        (1.20 * r, .74 * length, .88 * r))
                    line("tube", name + "_hinge", name,
                         (-1.22 * r, -length, .05 * r), (1.22 * r, -length, .05 * r), .43 * r)
                    put("shell", name + "_panel", name, (0, -.51 * length, 1.13 * r),
                        (.58 * r, .50 * length, .08 * r))
                elif style == "paw":
                    # Lift the enlarged soft volume dorsally to preserve key contact.
                    put("ball", name + "_fur", name, (0, -.5 * length, .35 * r),
                        (1.35 * r, .47 * length, .97 * r))
                    for sign in (-1, 1):
                        for t in (.30, .68):
                            line("tuft", name + "_toe_fuzz", name,
                                 (sign * 1.05 * r, -t * length, .35 * r),
                                 (sign * 1.48 * r, -t * length - .40 * r, .65 * r), .10 * r)
                if k == 3 and style in ("monster", "paw"):
                    claw_length = min(.018, max(.012, r * 1.65)) if style == "monster" else min(.006, max(.004, r * .62))
                    # An arched nail rooted behind the tip: at most .25r projects
                    # along the distal bone, which can point straight into a key.
                    claw_y = .25 * r - claw_length if style == "monster" else -.5 * claw_length
                    put("hook", name + "_claw", name, (0, claw_y, .55 * r),
                        (r * (.37 if style == "monster" else .22), claw_length, claw_length))
                    if style == "paw":
                        put("ball", name + "_plump_toe", name, (0, -.95 * r, .30 * r),
                            (1.40 * r, 1.10 * r, 1.06 * r))
                        pad_z = surface(name, 0, -.95 * r, back=False)
                        put("ball", name + "_bean", name, (0, -.95 * r, pad_z + .08 * r),
                            (.78 * r, .90 * r, .14 * r))

        if style == "monster":
            for m in range(1, 5):
                name = f"metacarpal{m}"
                r, length = radii[name], bones[name].length
                for t in (.25, .55, .80):
                    z = surface(name, 0, -t * length)
                    put("shell", name + "_scale", name, (0, -t * length, z + .03 * r),
                        (.34 * r, .23 * length, .17 * r))
                    line("spike", name + "_spine", name,
                         (0, -t * length, z + .12 * r), (0, -t * length + .12 * r, z + .80 * r), .22 * r)
        elif style == "robot":
            # Four panels leave visible seams and avoid a wide plate over the keys.
            plate_base = max((p.z for p in palm_points
                              if abs(p.x - px) <= palm_width and
                              .15 * palm_length <= p.y <= .90 * palm_length), default=top) + .02 * wr
            for j in range(4):
                x = px + (j - 1.5) * palm_width * .45
                put("shell", "back_panel", "wrist", (x, .53 * palm_length, plate_base + .10 * wr),
                    (.205 * palm_width, .64 * palm_length, .10 * wr), palm=True)
                for t in (.28, .78):
                    line("tube", "panel_mount", "wrist", (x, t * palm_length, top - .15 * wr),
                         (x, t * palm_length, plate_base), .08 * wr, palm=True)
                    put("ball", "rivet", "wrist", (x, t * palm_length, plate_base + .23 * wr),
                        (.065 * wr, .065 * wr, .045 * wr), palm=True)
            put("tube", "wrist_ring", "wrist", (0, -wrist.length + .15 * wr, 0),
                (1.12 * wr, .30 * wr, .86 * wr))
        elif style == "paw":
            put("ball", "back_fur", "wrist", (px, .45 * palm_length, top - .1 * wr),
                (.95 * palm_width, .47 * palm_length, .52 * wr), palm=True)
            # Three overlapping lobes plus a tapering lower lobe read as a heart/trefoil.
            for x, y, rx, ry in ((-.24, .56, .32, .24), (.24, .56, .32, .24),
                                  (0, .34, .40, .26)):
                pad_x, pad_y = px + x * palm_width, y * palm_length
                pad_z = surface("wrist", pad_x, pad_y, back=False, palm=True)
                put("ball", "main_pad", "wrist",
                    (pad_x, pad_y, pad_z - .025 * wr),
                    (rx * palm_width, ry * palm_length, .09 * wr), palm=True)
            for j in range(64):
                a = j * 2.399963
                radial = math.sqrt((j + .5) / 64)
                x = px + math.cos(a) * radial * palm_width * .91
                y = (.46 + math.sin(a) * radial * .39) * palm_length
                z = top + wr * (.12 + .24 * (1 - radial))
                line("tuft", "back_fuzz", "wrist", (x, y, z),
                     (x + math.cos(a) * wr * .24, y - .45 * wr, z + wr * .28), .065 * wr, palm=True)

        for name in ("lowerarm01", "lowerarm02"):
            b, r = bones[name], radii[name]
            length = b.length
            if style == "skeleton":
                for sign in (-1, 1):
                    put("shaft", name + ("_radius" if sign < 0 else "_ulna"), name,
                        (sign * .45 * r, -.5 * length, 0), (.20 * r, .97 * length, .18 * r))
                    for y in (-.98 * length, -.02 * length):
                        put("ball", name + "_end", name, (sign * .45 * r, y, 0),
                            (.24 * r, .28 * r, .23 * r))
            elif style == "monster":
                for j in range(6):
                    y = -length * (.09 + j * .16)
                    z = surface(name, 0, y)
                    put("shell", name + "_scale", name, (0, y, z + .02 * r),
                        (.81 * r, .13 * length, .20 * r))
                    line("spike", name + "_spine", name,
                         (0, y, z + .16 * r), (0, y + .20 * r, z + .82 * r), .24 * r)
                    for sign in (-1, 1):
                        z = surface(name, sign * .7 * r, y)
                        put("ball", name + "_wart", name, (sign * .7 * r, y, z + .02 * r),
                            (.26 * r, .29 * r, .26 * r))
            elif style == "robot":
                # Split at the twist-bone joint: every cylinder/rod follows its bone.
                for sign in (-1, 1):
                    x, z = sign * .86 * r, .66 * r
                    line("tube", name + "_piston_barrel", name,
                         (x, -.94 * length, z), (x, -.43 * length, z), .22 * r)
                    line("tube", name + "_piston_rod", name,
                         (x, -.46 * length, z), (x, -.04 * length, z), .11 * r)
                    for y in (-.90 * length, -.08 * length):
                        put("ball", name + "_mount", name, (x, y, z), (.27 * r, .22 * r, .26 * r))
                for t in (.18, .78):
                    put("shell", name + "_band", name, (0, -t * length, .10 * r),
                        (1.08 * r, .11 * length, .91 * r))
            elif style == "paw":
                put("ball", name + "_fluffy_cuff", name, (0, -.5 * length, .12 * r),
                    (1.17 * r, .57 * length, 1.10 * r))
                for j in range(12):
                    y = -length * (.025 + j * .086)
                    for k in range(24):
                        a = 2 * math.pi * (k + (j % 2) * .5) / 24
                        x, z = math.cos(a), math.sin(a)
                        line("tuft", name + "_cuff_fuzz", name,
                             (x * r * 1.05, y, z * r * 1.02 + .12 * r),
                             (x * r * 1.40, y - .52 * r, z * r * 1.36 + .12 * r), .085 * r)

    print(f"hand-style {style}: {len(made)} bone-parented objects, "
          f"{len(primitives)} shared meshes, "
          f"{sum(len(me.polygons) for me in primitives.values())} unique polygons")
    return made
