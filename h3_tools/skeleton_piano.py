"""Skeleton hands play the piano: H3 over a locked-off shot of a piano take, with the hand-looks recipe
(h3_tools/hand_looks.py), in chained clips so the look carries across clip boundaries.

    skeleton_piano.py passes RUN     Blender grey plate, normal pass and mask of every clip (skeleton geometry)
    skeleton_piano.py run    RUN     H3, clip after clip: clip k > 0 starts from clip k-1's last H3 frame
    skeleton_piano.py score  RUN     per clip: flow agreement with the normal pass inside the hand mask (motionmatch)
    skeleton_piano.py join   RUN     the clips back to back (+ the take's sound) -> ROOT/RUN/RUN.mp4, and a
                                     side-by-side with the Blender plate -> ROOT/RUN/RUN_sbs.mp4
    skeleton_piano.py final  RUN     a dense run (RUNS densify K): every K-th frame of the rendered clips (the take's
                                     own frames), the take's sound at speed -> ROOT/RUN/RUN_final.mp4; the lock check of
                                     it against the 24 fps reference run (RUNS ref, tolerance tol) -> lc_final.csv and
                                     overlay; a triptych (Blender | H3 | lock overlay) -> ROOT/RUN/RUN_triptych.mp4
    skeleton_piano.py run    RUN N   only the first N clips

RUN = a key of RUNS below: the edit (cameras.py --style still on the take), its first frame, the clip lengths
(17k+5 frames each), the view words for the prompt and the sound (the take's hands on the working piano).
The recipe: fl2va mode B, the grey plate WITH the look's geometry at video_strength 0.88, Fun ControlNet Union 2.0
(strength 1.0) on the Canny (20/60) of the normal pass of the same geometry; the look comes from that geometry and
the prompt. Every frame here is H3's; the plates and controls are Blender's.
"""
import json, math, os, pathlib, subprocess, sys, time

REPO = pathlib.Path(__file__).resolve().parent.parent
HOME = pathlib.Path.home()
ROOT = HOME / "h3/results/skeleton_piano"
COMFY_IN = HOME / "comfyui-h3-audio/input"
COMFY_OUT = HOME / "comfyui-h3-audio/output/ks_lipsync"
RUNNER = HOME / "h3audio/h3audio_run.py"
LEASE = HOME / "h3/gpu_lease.sh"
PY = str(HOME / "h3/mv_tools/.venv/bin/python")
LOOK = "skeleton"     # the look H3 paints (prompt only)
# Every H3 input (grey plate, normal pass, control, mask) is the stock MPFB2 hand: the project is the standard-hand
# overlay system, so the look must come from H3, never from look geometry in our renders. (Until 19 Oct 19:00 these
# runs used --hand-style skeleton, bone geometry in the plate and control: rejected by the user.) Look geometry may
# only be rendered as a measurement reference (RUNS "bone_ref": where the rig's bones are), never fed to H3.
HAND = "real"
SEED = 7
RUNS = {
    # Rachmaninoff, the opening (song 81.3 - 94.7 s), the overhead still shot of the C19 take
    "rach_open": {"edit": ROOT / "edit_c19_still.json", "f0": 993, "clips": [124, 124, 73],
                  "view": "Seen from directly above the keyboard",
                  "sound": HOME / "h3/hands_loop/action_20261017/suite_c19/cand/action_km02_audio/model.wav"},
    # the first 62 frames of clip 0 at twice the frame density (h3_tools/densify_take.py: an in-between pose after each
    # frame, originals kept), the sound slowed to half speed: H3's 4-frame latent spans half the motion. Lock test.
    "rach_open_x2": {"edit": ROOT / "edit_c19_x2.json", "f0": 0, "clips": [124],
                     "view": "Seen from directly above the keyboard", "sound": ROOT / "sound_x2_c0.wav"},
    # the close camera at 4x the frame density: one H3 latent frame (4 video frames) per original frame (T5)
    "rach_open_close_x4": {"edit": ROOT / "edit_c19_close_x4.json", "f0": 0, "clips": [124],
                           "view": "Seen from directly above the keyboard", "sound": ROOT / "sound_x4_c0.wav"},
    # "super size": the close camera at 1920x1088 (2.25x the H3 tokens; the same 124 frames as T2) (T6)
    "rach_open_close_hd": {"edit": ROOT / "edit_c19_close.json", "f0": 993, "clips": [124], "size": (1920, 1088),
                           "view": "Seen from directly above the keyboard",
                           "sound": HOME / "h3/hands_loop/action_20261017/suite_c19/cand/action_km02_audio/model.wav"},
    # production candidate: close camera, 4x density, the first 124 original frames (5.2 s) as 4 chained H3 clips;
    # `final` keeps every 4th frame (the originals), puts the take's sound back at speed and lock-checks the result
    # against the 24 fps close-camera reference (run rach_open_close: the same 124 original frames, take frames 993+)
    "rach_open_close_x4_full": {"edit": ROOT / "edit_c19_close_x4_full.json", "f0": 0, "clips": [124, 124, 124, 124],
                                "view": "Seen from directly above the keyboard", "sound": ROOT / "sound_x4_full.wav",
                                "densify": 4, "ref": "rach_open_close", "tol": 4.35},
    # T9a: the production run with every clip rendered on its own (no keyframe chain), same inputs and seed: chained
    # clips darkened ~3 grey levels a link (clip 0 25.7, clip 1 ~22, clip 2 ~19.8: keyframe re-draw + H3 settling)
    "rach_open_close_x4_indep": {"edit": ROOT / "edit_c19_close_x4_full.json", "f0": 0, "clips": [124, 124, 124, 124],
                                 "view": "Seen from directly above the keyboard", "sound": ROOT / "sound_x4_full.wav",
                                 "densify": 4, "ref": "rach_open_close", "tol": 4.35, "chain": False,
                                 "inputs_from": "rach_open_close_x4_full"},
    # T10: one 362-frame clip (the top of H3's trained 124-362 range) of the same dense take: 90 original frames
    # (3.8 s) with no seam and no chain. Unchained clips keep the look (T9a) but pop at every seam (frame diff 9.7 vs
    # 5.3 typical); chained clips darken ~3 grey levels a link. Plate/normal/mask frames are the x4_full ones.
    "rach_open_close_x4_long": {"edit": ROOT / "edit_c19_close_x4_full.json", "f0": 0, "clips": [362],
                                "view": "Seen from directly above the keyboard", "sound": ROOT / "sound_x4_full.wav",
                                "densify": 4, "ref": "rach_open_close", "tol": 4.35},
    # T12: T10 with block-sparse attention (BlockSparseAttention sol-attn, tau 1.3, dense for the first 20% of the
    # steps, text/audio rows exact): the kernel lever for bigger token counts, if it keeps the lock
    "rach_open_close_x4_long_sparse": {"edit": ROOT / "edit_c19_close_x4_full.json", "f0": 0, "clips": [362],
                                       "view": "Seen from directly above the keyboard", "sound": ROOT / "sound_x4_full.wav",
                                       "densify": 4, "ref": "rach_open_close", "tol": 4.35,
                                       "inputs_from": "rach_open_close_x4_long",
                                       "sparse": {"method": "sol-attn", "tau": 1.3, "start": 0.2, "verbose": True}},
    # the same at 1920x1088 (2.25x the H3 tokens, ~57 min a clip): super size on top of the density (T6b = its clip 0)
    "rach_open_close_x4_hd": {"edit": ROOT / "edit_c19_close_x4_full.json", "f0": 0, "clips": [124, 124, 124, 124],
                              "size": (1920, 1088), "view": "Seen from directly above the keyboard",
                              "sound": ROOT / "sound_x4_full.wav", "densify": 4, "ref": "rach_open_close", "tol": 4.35},
    # both: the close camera at twice the frame density (T4)
    "rach_open_close_x2": {"edit": ROOT / "edit_c19_close_x2.json", "f0": 0, "clips": [124],
                           "view": "Seen from directly above the keyboard", "sound": ROOT / "sound_x2_c0.wav"},
    # the same first clip with a locked-off overhead camera fitted to the hands only (~1.45x closer): lock test
    "rach_open_close": {"edit": ROOT / "edit_c19_close.json", "f0": 993, "clips": [124],
                        "view": "Seen from directly above the keyboard",
                        "sound": HOME / "h3/hands_loop/action_20261017/suite_c19/cand/action_km02_audio/model.wav"},
}


# The performance cut: the approved fingers-style edit of the C19 take (blockout/cameras.py --style fingers: low
# steadicam, orbit, across; cuts on downbeats). Seen from above, a key that goes down moves along the line of sight
# and never reads; from these cameras the key fronts drop. Each shot is one H3 clip (a cut hides any change of look
# between clips): density K = up to 4 so the shot fits H3's 362-frame trained range, frames past the cut keep the
# shot's camera and are dropped in `final`.
SHOW_EDIT = HOME / "h3/hands_loop/action_20261017/suite_c19/render/edit_show.json"
TAKE_SOUND = HOME / "h3/hands_loop/action_20261017/suite_c19/cand/action_km02_audio/model.wav"
SHOT_WORDS = {   # (view words, camera sentence) per shot type
    "steady_low": ("Seen from low beside the keyboard, close on the hands and the keys they press",
                   "The camera glides slowly beside the hands, low over the keys."),
    "orbit": ("Seen from beside the keyboard, close on the hands and the keys they press",
              "The camera circles slowly around the hands."),
    "across": ("Seen from across the keyboard, close on the hands and the keys they press",
               "The camera glides slowly along the keys."),
    "steady": ("Seen from beside the keyboard, close on the hands and the keys they press",
               "The camera follows the hands smoothly along the keys."),
    "both": ("Seen from beside the keyboard, both hands and the keys they press",
             "The camera glides slowly along the keyboard."),
    "overhead": ("Seen from above the keyboard, the hands and the keys they press",
                 "The camera follows the hands smoothly from above."),
    "descend": ("Seen from above the keyboard, the hands and the keys they press",
                "The camera comes down slowly toward the hands."),
    "rise": ("Seen from above the keyboard, the hands and the keys they press",
             "The camera rises slowly away from the hands."),
    "pov": ("Seen from the pianist's own point of view, looking down at the hands and the keys they press",
            "The camera moves gently with the pianist's head."),
}
# Rachmaninoff opening (take frames 993-1313, song 81.3-94.7 s): three shots of the show edit
SHOTS = [("rach_s1", 993, 88, "steady_low_R"), ("rach_s2", 1081, 146, "orbit_R"), ("rach_s3", 1227, 87, "across_L")]


def _grid(n):                                      # smallest 17k+5 >= n
    return 5 + 17 * max(0, -(-(n - 5) // 17))


def _show_shots(edit=None):
    """(first frame, frames, shot name) of every shot of the show edit (or another edit), in order."""
    F = json.loads((edit or SHOW_EDIT).read_text())["frames"]
    out, prev = [], None
    for i, fr in enumerate(F):
        s = fr["cam"].get("shot")
        if s != prev:
            out.append([i, 0, s]); prev = s
        out[-1][1] += 1
    return out


for _name, _f0, _n, _shot in SHOTS:
    _K = min(4, 362 // _n)
    while _grid(_K * _n) > 362:
        _K -= 1
    _L = _grid(_K * _n)
    _w = SHOT_WORDS[_shot.rsplit("_", 1)[0]]
    RUNS[_name] = {"edit": ROOT / f"edit_c19_show_{_name}.json", "f0": 0, "clips": [_L], "densify": _K,
                   "keep": _n, "take_f0": _f0, "shot": _shot, "view": _w[0], "camera_words": _w[1],
                   "sound": ROOT / f"sound_{_name}.wav", "tol": 3.0}


# The whole Knight's Suite (take C19, 8,415 frames, 350 s) on the show edit (the C19 performance cut's cameras): one
# H3 clip per shot at the take's own 24 fps (40 shots of 146-299 frames, all inside H3's 362), the overlay recipe
# (stock hand, pose control + keys, piano plate without hands, the skeleton first in the prompt). `suite` joins them.
SUITE, SIDE = [], []
# "work with what works, then fix what doesn't" (the user, 20 Oct): the same edit with the shots whose cameras fail on
# the overlay recipe re-shot on the camera types that work (h3_tools/recamera.py; same cuts and timing)
SIDE_EDIT = ROOT / "edit_show_side.json"


def _suite_runs(prefix, edit, out):
    for j, (f0, n, shot) in enumerate(_show_shots(edit)):
        name = f"{prefix}_s{j:02d}"
        w = SHOT_WORDS[shot.rsplit("_", 1)[0] if shot[-2:] in ("_L", "_R") else shot]
        RUNS[name] = {"edit": ROOT / f"edit_c19_show_{name}.json", "f0": 0, "clips": [_grid(n)], "densify": 1,
                      "keep": n, "take_f0": f0, "shot": shot, "view": w[0], "camera_words": w[1],
                      "sound": ROOT / f"sound_{name}.wav", "tol": 3.0, "control_kind": "posekeys",
                      "plate_kind": "plate_nohands", "prompt_lead": None, "source_edit": edit}
        out.append(name)


if SHOW_EDIT.exists():
    _suite_runs("suite", SHOW_EDIT, SUITE)
if SIDE_EDIT.exists():
    _suite_runs("side", SIDE_EDIT, SIDE)

# The suite's shot 1 ("both", from above) put the skeleton hands in front of the keys as open hands: seen from above,
# curled fingers point at the camera and their joints collapse into stubs. Probes on its first 124 frames:
# F = joints + the stock hand's outer outline; G = joints over all of the stock hand's edges (outline + knuckles).
if "suite_s01" in RUNS:
    RUNS["s01p"] = dict(RUNS["suite_s01"], clips=[124], keep=124)
    RUNS["s01p_outline"] = dict(RUNS["s01p"], inputs_from="s01p", control_kind="poseoutline")
    RUNS["s01p_edges"] = dict(RUNS["s01p"], inputs_from="s01p", control_kind="poseedges")
    # F (outline) puts the hands on the rig, curled on the keys, but paints them flesh. The side view at 24 fps:
    RUNS["s02p"] = dict(RUNS["suite_s02"], clips=[124], keep=124)
    # s02p (steady, ~40 deg up, 24 fps): ghostly smeared bones. Which side cameras hold at 24 fps:
    for _j in (4, 5, 6):                           # steady_low_R, orbit_R, across_L
        RUNS[f"s{_j:02d}p"] = dict(RUNS[f"suite_s{_j:02d}"], clips=[124], keep=124)
    # s04p (steady_low at 24 fps) lost the hands: stringy bones along the keys; the same shot at 4x (std_s1 D) held.
    # The rig's joints move too far inside one H3 latent frame. Does 2x hold? (the same start, 62 take frames)
    RUNS["s04x2p"] = dict(RUNS["suite_s04"], densify=2, clips=[124], keep=62,
                          edit=ROOT / "edit_c19_show_s04x2p.json", sound=ROOT / "sound_s04x2p.wav")

# probe: the first 124 dense frames of shot 1 (31 take frames): do the keys go down in H3 from this camera?
# (rendered on skeleton geometry before the standard-hand rule; its passes now serve only as the bone reference)
RUNS["rach_s1_probe"] = dict(RUNS["rach_s1"], clips=[124], keep=31, hand_ref="skeleton")
# the same probe on the stock hand: the overlay system. bone_ref = where the rig's bones are (measurement only)
RUNS["std_s1_probe"] = dict(RUNS["rach_s1"], clips=[124], keep=31, bone_ref="rach_s1_probe")
# std_s1_probe painted flesh hands on the stock-hand outline (hands 53%, bones 14%): the control sets the category.
# A: release the ControlNet after the first 40% of the steps (layout early, the look free late)
RUNS["std_s1_cend04"] = dict(RUNS["std_s1_probe"], inputs_from="std_s1_probe", control_end=0.4)
# B: the overlay control: the rig's joints as an OpenPose hand skeleton + the keys' edges with the hands blanked
# (h3_tools/pose_control.py on blender_piano.py --pose2d of the stock hand); nothing in it says what the hand looks like
RUNS["std_s1_pose"] = dict(RUNS["std_s1_probe"], inputs_from="std_s1_probe", control_kind="posekeys")
# C: B + a plate of the piano alone (blender_piano.py --hide hands): no hand look anywhere in H3's inputs
RUNS["std_s1_pose_nh"] = dict(RUNS["std_s1_pose"], plate_kind="plate_nohands")
# A, B, C all painted flesh hands (C even with no hand anywhere in H3's inputs): H3's prior for hands at a piano wins
# over a skeleton look named mid-prompt. D: C + the look first, in so many words. E: D + prompt guidance 4 with a
# negative (CFGGuider; H3 ships for guidance 1).
SKEL_LEAD = ("Two skeleton hands made only of bare ivory-white bone, with no skin, no flesh and no muscle: every finger "
             "a chain of separate bones with knobbly joints, the bony wrists and the bare radius and ulna of the forearms.")
RUNS["std_s1_pose_nh_lead"] = dict(RUNS["std_s1_pose_nh"], prompt_lead=SKEL_LEAD)
for _name in SUITE + SIDE:
    RUNS[_name]["prompt_lead"] = SKEL_LEAD
# D works: H3 paints skeleton hands over the stock hand, bones on the rig (63% of the bone geometry's edges within 3 px,
# the flesh-hand tests 33%). The overlay recipe: stock-hand pose control + keys, a plate without hands, the look first.
LOOK_LEADS = {
    "skeleton": SKEL_LEAD,
    "monster": ("Two monster hands, not human: rough scaly creature skin, hooked black talons and knobbly clawed fingers, "
                "rows of short spikes on the backs of the hands and jagged spines along the scaly forearms."),
    "robot": ("Two robot hands made entirely of brushed metal, no skin anywhere: hard metal finger segments with a hinge pin "
              "at every joint, riveted panels on the backs of the hands, pistons along the mechanical forearms."),
    "paw": ("Two fluffy kitty-cat paws, not human hands: soft dense cat fur over rounded furry toes with tiny curved claws, "
            "pink toe beans, furry cat forearms."),
    "real": "Two human pianist's hands: natural skin, visible knuckles and tendons, short clean fingernails.",
}
for _look in LOOK_LEADS:                           # the five looks on the same probe, the same stock-hand inputs
    RUNS[f"ov_s1_{_look}"] = dict(RUNS["std_s1_pose_nh"], look=_look, prompt_lead=LOOK_LEADS[_look])
# paw v1 painted human hands with fur cuffs (the paw shape fights a human finger pose); v2: fur over every finger
RUNS["ov_s1_paw2"] = dict(RUNS["ov_s1_paw"], prompt_lead=(
    "Two furry cat-paw hands, no bare skin anywhere: long fingers completely covered in thick fluffy orange tabby cat "
    "fur, rounded furry fingertips with tiny curved claws and pink toe beans, fluffy tabby cat fur all along the forearms."))
RUNS["std_s1_pose_nh_cfg"] = dict(RUNS["std_s1_pose_nh_lead"], cfg=4.0,
                                  negative="human hands, skin, flesh, fingernails, realistic human fingers, gloves")


def nohands(run):
    """The piano without hands for every clip of a run (plate_nohands): H3's video stream, hands from the pose control."""
    R = RUNS[run]
    out = render(run, "plate_nohands", ["--hide", "hands"])
    for k, (f0, n) in enumerate(clip_frames(run)):
        encode(out, COMFY_IN / cin(run, k, "plate_nohands"), f0, n, size=R.get("size", (1280, 704)))
    print("plate_nohands", run, "ok")


# The three shots of the Rachmaninoff opening on the overlay recipe (stock hand; the skeleton only in the prompt).
# bone_ref = the shot's skeleton-geometry passes (rendered before the rule), measurement only.
for _s in ("rach_s1", "rach_s2", "rach_s3"):
    RUNS["ov_" + _s] = dict(RUNS[_s], control_kind="posekeys", plate_kind="plate_nohands", prompt_lead=SKEL_LEAD,
                            bone_ref=_s)


# The user (20 Oct): render the Rachmaninoff opening now, at 4x (2x paints half-flesh hands, 24 fps loses them).
# Shot 2 (orbit, 146 take frames) at 4x is two chained 294-frame clips (the second opens on the first's last H3 frame).
# ov_rach_s1 (362 frames) painted dark flesh hands where its 124-frame probe (D, same camera, frames, seed) painted
# skeletons: on a long clip H3 drifts back to its default hands. Anchor the look: D's own first frame (H3's skeleton
# at the same take frame 993) as the keyframe of the long clip.
RUNS["ov_rach_s1_kf"] = dict(RUNS["ov_rach_s1"], inputs_from="ov_rach_s1",
                             first_image="band/skeleton/kf_rach_s1_skeleton.png")
# ov_rach_s1_kf: frame 0 the skeleton keyframe, then the hands dissolve into dark ghosts with the pose lines drawn
# as glowing wires. Long clips don't hold on this recipe; every clip that worked was 124 frames. Shot 1 as three
# chained 124-frame clips: clip 0 = probe D (same dense frames 0-123), clips 1-2 open on the previous clip's last frame.
RUNS["ov_rach_s1_ch"] = dict(RUNS["ov_rach_s1"], clips=[124, 124, 124], edit=ROOT / "edit_c19_show_ov_rach_s1_ch.json",
                             sound=ROOT / "sound_ov_rach_s1_ch.wav", bone_ref=None)
# The chain holds the skeleton across clips (seams smooth); each link settles ~17% darker (24 -> 20 -> 17), reset at
# the next camera cut. Shots 2 (orbit, 146 take frames) and 3 (across, 87) as chained 124-frame clips at 4x.
RUNS["ov_rach_s2_ch"] = dict(RUNS["ov_rach_s2"], densify=4, clips=[124] * 5, keep=146, bone_ref=None,
                             edit=ROOT / "edit_c19_show_ov_rach_s2_ch.json", sound=ROOT / "sound_ov_rach_s2_ch.wav")
RUNS["ov_rach_s3_ch"] = dict(RUNS["ov_rach_s3"], clips=[124] * 3, bone_ref=None,
                             edit=ROOT / "edit_c19_show_ov_rach_s3_ch.json", sound=ROOT / "sound_ov_rach_s3_ch.wav")
# Shot 2's orbit ends just behind the right hand (clips 2-4, ~40 deg up, 0.39 m): H3 painted the joints of fingers
# pointing away from the camera as long thin wires, the prompt still saying "beside". Probes on clip 3 alone,
# unanchored: V = the view in words from where the camera is (view_words); VW = V + the joint lines as wide as half a
# bone through the lens (pose_control.py --width). The high cameras failed at 24 fps but were never tried at 4x:
# S01X4 = the suite's shot 1 ("both", from above) at 4x, its first 124 dense frames, with V.
RUNS["s2c3_v"] = dict(RUNS["ov_rach_s2_ch"], f0=RUNS["ov_rach_s2_ch"]["f0"] + 3 * 124, clips=[124], view_auto=True,
                      chain=False)
RUNS["s2c3_vw"] = dict(RUNS["s2c3_v"], control_kind="posekeysw")
# s2c3_v (view words alone): full skeleton hands seen from behind (radius and ulna, carpals, metacarpals, fingers on
# the keys) where the chain painted wires. The opening's bad clips again with view_words: shot 2 clips 2-4 chained on
# from the chain's clip 1 (clips 0-1 adopted), shot 3 all three clips.
# Every chained clip settles ~10% darker within its first two latent frames (shot 1: 39.2 -> 37.4 -> 34.9 across the
# seam, a visible dim step every 1.3 s). Overlap probe: shot 1 with each later clip opening 9 frames (3 latent frames)
# before the previous one ends, those frames the previous clip's H3 frames held clean (lock head), dropped in final.
RUNS["ov_rach_s1_ov"] = dict(RUNS["ov_rach_s1_ch"], overlap=9)   # same prompt as the chain: a clean A/B
RUNS["ov_rach_s2_cv"] = dict(RUNS["ov_rach_s2_ch"], inputs_from="ov_rach_s2_ch", view_auto=True)
RUNS["ov_rach_s3_cv"] = dict(RUNS["ov_rach_s3_ch"], inputs_from="ov_rach_s3_ch", view_auto=True)
if "suite_s01" in RUNS:
    RUNS["s01x4p"] = dict(RUNS["suite_s01"], densify=4, clips=[124], keep=31, view_auto=True,
                          edit=ROOT / "edit_c19_show_s01x4p.json", sound=ROOT / "sound_s01x4p.wav")
RUNS["ov_rach_s2x4"] = dict(RUNS["ov_rach_s2"], densify=4, clips=[294, 294], keep=146,
                            edit=ROOT / "edit_c19_show_ov_rach_s2x4.json", sound=ROOT / "sound_ov_rach_s2x4.wav")


def overlay_inputs(run):
    """Everything the overlay recipe feeds H3, all from the stock hand: passes (plate, normal, mask; the normal pass gives
    the keys' edges), the piano without hands, the rig's joints through the camera (blender_piano.py --pose2d), and the
    control (pose_control.py: OpenPose hand skeleton + keys, hands blanked)."""
    R = RUNS[run]
    if "keep" in R:
        shot_edit(run)
    passes(run)
    nohands(run)
    a0, n = R["f0"], sum(R["clips"])
    W, H = R.get("size", (1280, 704))
    pj = ROOT / run / "pose2d.json"
    if not pj.exists():
        subprocess.run(["blender", "-b", "--factory-startup", "-P", str(REPO / "blockout/blender_piano.py"), "--",
                        str(R["edit"]), str(ROOT / run / "pose_tmp"), "--view", "edit", "--frames", f"{a0}:{a0 + n}",
                        "--hand-style", HAND, "--res", f"{W}x{H}", "--pose2d", str(pj)], cwd=REPO,
                       env=dict(os.environ, RIG_SLEEVE="0.06"), check=True, capture_output=True)
    P = json.loads(pj.read_text())
    E = json.loads(R["edit"].read_text())["frames"]
    for k, (f0, n) in enumerate(clip_frames(run)):
        sub = ROOT / run / f"pose2d_c{k}.json"
        sub.write_text(json.dumps({"size": P["size"], "frames": {f: dict(P["frames"][f], lens=E[int(f)]["cam"]["lens"])
                                                                 for f in sorted(P["frames"], key=int)
                                                                 if f0 <= int(f) < f0 + n}}))
        ctl = R.get("control_kind", "posekeys")
        flag = {"posekeys": [], "posekeysd": ["--depth"], "posekeysw": ["--width"]}.get(ctl, [])
        subprocess.run([PY, str(REPO / "h3_tools/pose_control.py"), str(sub), str(ROOT / run / "normal"),
                        str(ROOT / run / "mask"), str(COMFY_IN / cin(run, k, ctl if flag else "posekeys"))] + flag,
                       check=True)
    print("overlay inputs", run, "ok")


def controls(run, kind):
    """The pose control of another flavour (posekeysw: --width, posekeysd: --depth) for every clip of a run's inputs, from
    its pose2d_c{k}.json (each frame's lens from the run's edit)."""
    i = src(run); R = RUNS[i]
    E = json.loads(R["edit"].read_text())["frames"]
    flag = {"posekeysw": "--width", "posekeysd": "--depth"}[kind]
    for k in range(len(R["clips"])):
        sub = ROOT / i / f"pose2d_c{k}.json"
        P = json.loads(sub.read_text())
        for f, fr in P["frames"].items():
            fr["lens"] = E[int(f)]["cam"]["lens"]
        sub.write_text(json.dumps(P))
        subprocess.run([PY, str(REPO / "h3_tools/pose_control.py"), str(sub), str(ROOT / i / "normal"), str(ROOT / i / "mask"),
                        str(COMFY_IN / cin(i, k, kind)), flag], check=True)


def adopt(run, frm, upto):
    """Clips 0..upto-1 of run are frm's (H3 output and its PNG frames, so the chain carries on from frm's clip upto-1)."""
    for k in range(upto):
        if not out_mp4(run, k).exists():
            subprocess.run(["cp", str(out_mp4(frm, k)), str(out_mp4(run, k))], check=True)
        if (COMFY_OUT / tag(frm, k)).is_dir() and not (COMFY_OUT / tag(run, k)).exists():
            subprocess.run(["cp", "-r", str(COMFY_OUT / tag(frm, k)), str(COMFY_OUT / tag(run, k))], check=True)


def shot_edit(run):
    """The dense edit and slowed sound of one shot: densify_take on the show edit (camera interpolated inside the
    shot), frames past the cut hold the shot's last camera; the take's sound from the shot's start, slowed K times."""
    R = RUNS[run]; K, n, f0, L = R["densify"], R["keep"], R["take_f0"], sum(R["clips"])
    n_ext = -(-L // K)
    if not R["edit"].exists():
        subprocess.run([PY, str(REPO / "h3_tools/densify_take.py"), str(R.get("source_edit", SHOW_EDIT)), str(R["edit"]), str(f0), str(n_ext),
                        str(K)], check=True)
        d = json.loads(R["edit"].read_text())
        held = d["frames"][(n - 1) * K]["cam"]
        assert held.get("shot") == R["shot"], (held.get("shot"), R["shot"])
        for fr in d["frames"][n * K:]:
            fr["cam"] = dict(held)
        R["edit"].write_text(json.dumps(d))
    if not R["sound"].exists():
        tempo = {1: "anull", 2: "atempo=0.5", 3: "atempo=0.5,atempo=0.6666667", 4: "atempo=0.5,atempo=0.5"}[K]
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{f0 / 24.0:.4f}", "-t", f"{n_ext / 24.0 + 0.1:.4f}",
                        "-i", str(TAKE_SOUND), "-af", tempo, "-ac", "2", "-ar", "48000", str(R["sound"])], check=True)
    print(run, R["shot"], "K", K, "frames", L, "originals", n, "of", n_ext)


def clip_frames(run):
    """(first dense frame, length) of each clip; with "overlap" O every clip after the first opens O frames before the
    previous one ends (those O frames are the previous clip's H3 frames, held; final drops them)."""
    R = RUNS[run]; out, f = [], R["f0"]
    for n in R["clips"]:
        out.append((f, n)); f += n - R.get("overlap", 0)
    return out


def overlap_plate(run, k):
    """Clip k's video stream with an overlap: its first O frames are clip k-1's last O H3 frames (lossless PNGs), the
    rest the hand-free plate. With those latent frames held clean (job "lock" head), clip k continues clip k-1 from
    real context (3 latent frames) instead of one keyframe, after which H3 settled ~10% darker within two latent frames."""
    import cv2
    O = RUNS[run]["overlap"]; i = src(run)
    pngs = sorted((COMFY_OUT / tag(run, k - 1)).glob("frame_*.png"))[-O:]
    plate = str(COMFY_IN / cin(i, k, RUNS[run].get("plate_kind", "plate")))
    cap = cv2.VideoCapture(plate); frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(im)
    H, W = frames[0].shape[:2]
    for j, pth in enumerate(pngs):
        frames[j] = cv2.resize(cv2.imread(str(pth)), (W, H), interpolation=cv2.INTER_AREA)
    dst = COMFY_IN / cin(run, k, "plate_ov")
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "24",
                             "-i", "-", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    for im in frames:
        proc.stdin.write(im.tobytes())
    proc.stdin.close(); proc.wait()
    return pngs[0]


def render(run, kind, extra, procs=4):
    """Blender frames of the whole run (all clips) of one pass into ROOT/run/kind/."""
    R = RUNS[run]
    out = ROOT / run / kind
    out.mkdir(parents=True, exist_ok=True)
    a0, n = R["f0"], sum(R["clips"])
    if len(list(out.glob("f_*.png"))) >= n:
        return out
    env = dict(os.environ, RIG_SLEEVE="0.06")
    step = -(-n // procs)
    W, H = R.get("size", (1280, 704))
    for attempt in range(3):                   # Blender's GL can segfault next to a running H3: rerun missing parts
        ps = []
        for k in range(procs):
            a, b = a0 + k * step, min(a0 + (k + 1) * step, a0 + n)
            if a >= b or all((out / f"f_{f:05d}.png").exists() for f in range(a, b)):
                continue
            cmd = ["blender", "-b", "--factory-startup", "-P", str(REPO / "blockout/blender_piano.py"), "--",
                   str(R["edit"]), str(out), "--view", "edit", "--frames", f"{a}:{b}", "--hand-style", R.get("hand_ref", HAND),
                   "--res", f"{W}x{H}"] + extra
            ps.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=open(out / f"blender_{k}.log", "w"),
                                       stderr=subprocess.STDOUT))
        for p in ps:
            p.wait()
        if len(list(out.glob("f_*.png"))) >= n:
            break
    got = len(list(out.glob("f_*.png")))
    assert got >= n, f"{out}: {got} frames"
    return out


def encode(src, dst, f0, n, gray=False, crf=14, size=(1280, 704)):
    vf = f"scale={size[0]}:{size[1]}" + (",format=gray" if gray else "")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-start_number", str(f0), "-i", f"{src}/f_%05d.png",
                    "-frames:v", str(n), "-vf", vf, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", str(dst)],
                   check=True)


def canny(src, dst, f0, n, size=(1280, 704)):
    import cv2
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{size[0]}x{size[1]}", "-r", "24",
                          "-i", "-", "-c:v", "libx264", "-crf", "8", "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    for f in range(f0, f0 + n):
        im = cv2.resize(cv2.imread(str(src / f"f_{f:05d}.png")), tuple(size), interpolation=cv2.INTER_AREA)
        p.stdin.write(cv2.Canny(cv2.GaussianBlur(im, (3, 3), 0), 20, 60).tobytes())
    p.stdin.close(); p.wait()


def cin(run, k, kind, ext="mp4"):
    return f"band/skeleton/{run}_c{k}_{kind}.{ext}"


def passes(run):
    R = RUNS[run]
    (COMFY_IN / "band/skeleton").mkdir(parents=True, exist_ok=True)
    plate = render(run, "plate", [])
    normal = render(run, "normal", ["--pass", "normal"])
    mask = render(run, "mask", ["--pass", "mask"])
    for k, (f0, n) in enumerate(clip_frames(run)):
        sz = R.get("size", (1280, 704))
        encode(plate, COMFY_IN / cin(run, k, "plate"), f0, n, size=sz)
        encode(normal, ROOT / run / f"c{k}_normal.mp4", f0, n, crf=12, size=sz)
        encode(mask, ROOT / run / f"c{k}_mask.mp4", f0, n, gray=True, crf=10, size=sz)
        canny(normal, COMFY_IN / cin(run, k, "ncanny"), f0, n, size=sz)
        t0 = f0 / 24.0                             # seconds into the take (model.wav starts at the take's frame 0)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.4f}", "-t", f"{n / 24.0:.4f}", "-i", str(R["sound"]),
                        "-ac", "2", "-ar", "48000", str(COMFY_IN / cin(run, k, "sound", "wav"))], check=True)
    print("passes", run, "ok")


def view_words(run, k):
    """Clip k's view in words, from where its camera really is (circular mean over the clip of the camera's direction
    from the point it looks at; the player sits on -y, the keys run along x). The shot type's words are one sentence for
    the whole shot: an orbit that starts beside the keyboard and ends just behind the right hand kept saying "beside",
    and H3 painted the joints of fingers pointing away from the camera as long thin wires."""
    R = RUNS[run]
    F = json.loads(R["edit"].read_text())["frames"]
    f0, n = clip_frames(run)[k]
    ex = ey = 0.0; el = []
    for fr in F[f0:f0 + n]:
        c = fr["cam"]; v = [c["pos"][i] - c["look"][i] for i in range(3)]
        h = math.hypot(v[0], v[1])
        el.append(math.degrees(math.atan2(v[2], h))); ex += v[0] / max(h, 1e-6); ey += v[1] / max(h, 1e-6)
    e = sorted(el)[len(el) // 2]
    az = math.degrees(math.atan2(ey, ex))                          # 0: the treble end (+x), -90: the player's side
    d = abs((az + 90 + 180) % 360 - 180)                           # 0 = straight behind the hands, 180 = facing them
    side = R.get("shot", "")[-2:]
    hand = {"_R": "right hand", "_L": "left hand"}.get(side, "hands")
    end = "the treble" if ex > 0 else "the bass"
    high = " high" if e > 45 else (" low" if e < 22 else "")
    if e >= 60:
        return ("Seen from high above the keyboard, looking down on the backs of the hands, the fingers curled down onto "
                "the keys they press")
    if d <= 30:
        back = "the wrists and the backs of the hands" if hand == "hands" else "the wrist and the back of the hand"
        return (f"Seen from just behind and above the pianist's {hand}, looking over {back}, "
                "the fingers reaching away from the camera and curling down onto the keys they press")
    if d <= 55:
        return (f"Seen from{high} behind the {hand}, from {end} end of the keyboard, looking along the keys, close on "
                "the fingers and the keys they press")
    if d <= 125:
        return f"Seen from{high} beside the keyboard at {end} end, close on the hands and the keys they press"
    if d <= 150:
        return (f"Seen from{high} across the keyboard, from the far side of the keys, the fingertips toward the camera "
                "and the keys they press")
    return ("Seen from across the piano, facing the pianist, the fingertips coming toward the camera onto the keys "
            "they press")


def prompt(run, k=None):
    P = json.loads((REPO / "h3_tools/hand_look_prompts.json").read_text())
    view = view_words(run, k) if k is not None and RUNS[run].get("view_auto") else RUNS[run]["view"]
    t = P["base"].format(**P["looks"][RUNS[run].get("look", LOOK)]).replace("Seen from high above and behind the player", view)
    if RUNS[run].get("camera_words"):              # a moving shot: not locked off
        t = t.replace("One continuous locked-off shot", "One continuous shot").replace(
            "The camera does not move.", RUNS[run]["camera_words"])
    if RUNS[run].get("prompt_lead"):               # the look first, in so many words
        t = RUNS[run]["prompt_lead"] + " " + t
    return t


def tag(run, k):
    return f"sk_{run}_c{k}"


def out_mp4(run, k):
    return COMFY_OUT / f"{tag(run, k)}_00001_.mp4"


def src(run):
    """The run whose passes (plate, control, sound, normal, mask) this run uses (RUNS inputs_from; default itself)."""
    return RUNS[run].get("inputs_from", run)


def job(run, k, first=None):
    n = clip_frames(run)[k][1]
    W, H = RUNS[run].get("size", (1280, 704))
    i = src(run)
    j = {"steps": 20, "width": W, "height": H, "length": n, "scheduler": "beta", "sampler": "res_multistep",
         "seed": SEED, "audio": cin(i, k, "sound", "wav"), "prompt": prompt(run, k), "mode": "B",
         "video": cin(i, k, RUNS[run].get("plate_kind", "plate")), "video_strength": 0.88,
         "control": {"video": cin(i, k, RUNS[run].get("control_kind", "ncanny")), "strength": RUNS[run].get("control_strength", 1.0),
                     "end": RUNS[run].get("control_end", 1.0)}, "tag": tag(run, k)}
    for key in ("cfg", "negative"):                # prompt guidance (h3audio_run.py "cfg")
        if RUNS[run].get(key):
            j[key] = RUNS[run][key]
    if RUNS[run].get("sparse"):                    # block-sparse attention (h3audio_run.py "sparse")
        j["sparse"] = RUNS[run]["sparse"]
    if RUNS[run].get("overlap") and k:             # overlap: the previous clip's last frames held as context
        O = RUNS[run]["overlap"]
        j["video"] = cin(run, k, "plate_ov")
        j["lock"] = {"head": 1 + (O - 1) // 4, "anchor": 0.0, "free": j["video_strength"]}
    if first:                                      # chained: this clip opens on the previous clip's last H3 frame
        j["image"] = first
    else:
        j["no_first"] = True
    (ROOT / run / "cfg").mkdir(parents=True, exist_ok=True)
    p = ROOT / run / "cfg" / f"{tag(run, k)}.json"
    p.write_text(json.dumps(j, indent=1))
    return p


def last_frame(run, k, dst):
    """Clip k's last H3 frame, lossless: the PNG the runner saves beside the mp4 (the mp4's last frame is ~0.6 grey
    levels darker after the yuv420 round trip, and chained clips compound it); the mp4 if no PNGs."""
    pngs = sorted((COMFY_OUT / tag(run, k)).glob("frame_*.png"))
    if pngs:
        subprocess.run(["cp", str(pngs[-1]), str(dst)], check=True)
        return
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-sseof", "-0.1", "-i", str(out_mp4(run, k)), "-frames:v", "1",
                    "-update", "1", str(dst)], check=True)


def run_all(run, need=int(os.environ.get("H3_NEED_GB", 30)), poll=180, upto=None):
    for k in range(len(RUNS[run]["clips"]) if upto is None else upto):
        if out_mp4(run, k).exists():
            print("exists:", out_mp4(run, k)); continue
        first = RUNS[run].get("first_image") if k == 0 else None   # a keyframe that anchors the look (an H3 frame)
        if k and RUNS[run].get("chain", True):          # chain = False: every clip on its own (no keyframe), like clip 0
            first = cin(run, k, "first", "png")
            if RUNS[run].get("overlap"):
                subprocess.run(["cp", str(overlap_plate(run, k)), str(COMFY_IN / first)], check=True)
            else:
                last_frame(run, k - 1, COMFY_IN / first)
        p = job(run, k, first)
        while True:
            r = subprocess.run(["bash", str(LEASE), "take", "h3-skeleton", f"H3 skeleton piano {run} clip {k}", str(need)],
                               capture_output=True, text=True)
            if r.returncode == 0:
                break
            print(time.strftime("%H:%M:%S"), "waiting:", r.stdout.strip(), flush=True)
            time.sleep(poll)
        try:
            print(time.strftime("%H:%M:%S"), "H3", tag(run, k), flush=True)
            subprocess.run(["python3", str(RUNNER), str(p)], check=True)
        finally:
            subprocess.run(["bash", str(LEASE), "release", "h3-skeleton"], capture_output=True)
        if not out_mp4(run, k).exists():                # the runner exits 0 when the container goes down mid-clip
            sys.exit(f"{time.strftime('%H:%M:%S')} no output for {tag(run, k)} (container down?); stopping the chain")
        score(run, k)


def score(run, k):
    ref, mask = ROOT / src(run) / f"c{k}_normal.mp4", ROOT / src(run) / f"c{k}_mask.mp4"
    plate = COMFY_IN / cin(src(run), k, "plate")
    mm = subprocess.check_output([PY, str(REPO / "h3_tools/motionmatch.py"), str(ref), str(out_mp4(run, k)), str(mask),
                                  f"{run} c{k} vs normal"]).decode().strip()
    mp = subprocess.check_output([PY, str(REPO / "h3_tools/motionmatch.py"), str(plate), str(out_mp4(run, k)), str(mask),
                                  f"{run} c{k} vs plate"]).decode().strip()
    with open(ROOT / run / "scores.txt", "a") as fh:
        fh.write(mm + "\n" + mp + "\n")
    print(mm); print(mp)


def join(run):
    R = RUNS[run]; d = ROOT / run
    n = len(R["clips"])
    lst = d / "clips.txt"
    lst.write_text("".join(f"file '{out_mp4(run, k)}'\n" for k in range(n)))
    t0, dur = R["f0"] / 24.0, sum(R["clips"]) / 24.0
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-ss", f"{t0:.4f}",
                    "-t", f"{dur:.4f}", "-i", str(R["sound"]), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(d / f"{run}.mp4")], check=True)
    plates = d / "plates.txt"
    plates.write_text("".join(f"file '{COMFY_IN / cin(run, k, 'plate')}'\n" for k in range(n)))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(plates), "-i", str(d / f"{run}.mp4"),
                    "-filter_complex", "[0:v]scale=640:352[a];[1:v]scale=640:352[b];[a][b]hstack", "-map", "1:a",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
                    str(d / f"{run}_sbs.mp4")], check=True)
    print("joined", d / f"{run}.mp4")


def decimate(src, dst, K, n, gray=False):
    vf = f"select='not(mod(n\\,{K}))',setpts=N/24/TB" + (",format=gray" if gray else "")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-r", "24", "-frames:v", str(n),
                    "-c:v", "libx264", "-crf", "10", "-pix_fmt", "yuv420p", str(dst)], check=True)


def _cat(paths, dst, trim=0):
    """paths joined; trim: drop the first trim frames of every path after the first (a run's clip overlap)."""
    if len(paths) == 1:
        return paths[0]
    if trim:
        ins = sum((["-i", str(p)] for p in paths), [])
        fc = "".join(f"[{i}:v]" + (f"trim=start_frame={trim},setpts=PTS-STARTPTS" if i else "null") + f"[v{i}];"
                     for i in range(len(paths)))
        fc += "".join(f"[v{i}]" for i in range(len(paths))) + f"concat=n={len(paths)}:v=1:a=0[v]"
        subprocess.run(["ffmpeg", "-v", "error", "-y"] + ins + ["-filter_complex", fc, "-map", "[v]", "-c:v", "libx264",
                        "-crf", "10", "-pix_fmt", "yuv420p", str(dst)], check=True)
        return dst
    lst = dst.with_suffix(".txt")
    lst.write_text("".join(f"file '{p}'\n" for p in paths))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264", "-crf", "10",
                    "-pix_fmt", "yuv420p", str(dst)], check=True)
    return dst


def final_shot(run):
    """A shot run: its H3 clip's every K-th frame up to the cut, the take's sound at speed, the lock check against the
    shot's own passes taken the same way (dense frame K*i is take frame take_f0 + i exactly), and the triptych."""
    R = RUNS[run]; d = ROOT / run; K, n = R["densify"], R["keep"]
    out = d / f"{run}_final.mp4"
    video = out_mp4(run, 0)
    O = R.get("overlap", 0)
    if len(R["clips"]) > 1 and O:                  # overlapping clips: each later clip without its held frames
        video = _cat([out_mp4(run, k) for k in range(len(R["clips"]))], d / "clips_joined.mp4", trim=O)
    elif len(R["clips"]) > 1:                      # a shot too long for one clip at K: its clips back to back
        lst = d / "clips.txt"
        lst.write_text("".join(f"file '{out_mp4(run, k)}'\n" for k in range(len(R["clips"]))))
        video = d / "clips_joined.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-an", "-c:v", "libx264",
                        "-crf", "12", "-pix_fmt", "yuv420p", str(video)], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-ss", f"{R['take_f0'] / 24.0:.4f}",
                    "-t", f"{n / 24.0:.4f}", "-i", str(TAKE_SOUND), "-map", "0:v", "-map", "1:a",
                    "-vf", f"select='not(mod(n\\,{K}))',setpts=N/24/TB", "-r", "24", "-frames:v", str(n),
                    "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(out)],
                   check=True)
    d.mkdir(parents=True, exist_ok=True)
    nc = len(R["clips"])
    for kind, gray in (("normal", False), ("mask", True)):
        decimate(_cat([ROOT / src(run) / f"c{k}_{kind}.mp4" for k in range(nc)], d / f"all_{kind}.mp4", O),
                 d / f"ref_{kind}.mp4", K, n, gray)
    decimate(_cat([COMFY_IN / cin(src(run), k, "plate") for k in range(nc)], d / "all_plate.mp4", O), d / "ref_plate.mp4", K, n)
    lc = subprocess.run([PY, str(REPO / "h3_tools/lockcheck.py"), str(out), str(d / "ref_normal.mp4"), str(d / "ref_mask.mp4"),
                         str(d / "lc_final"), "--tol", str(R["tol"])], check=True, capture_output=True, text=True).stdout.strip()
    if R.get("bone_ref"):                          # are H3's bones where the rig's bones are? (reference only)
        bd = ROOT / R["bone_ref"]
        for kind, gray in (("normal", False), ("mask", True)):
            decimate(bd / f"c0_{kind}.mp4", d / f"bone_{kind}.mp4", K, n, gray)
        bones = subprocess.run([PY, str(REPO / "h3_tools/lockcheck.py"), str(out), str(d / "bone_normal.mp4"),
                                str(d / "bone_mask.mp4"), str(d / "lc_bones"), "--tol", str(R["tol"])],
                               check=True, capture_output=True, text=True).stdout.strip()
        with open(d / "scores.txt", "a") as fh:
            fh.write(f"bones (rig bone geometry, reference only): {bones}\n")
        print("bones:", bones)
    ceil = subprocess.run([PY, str(REPO / "h3_tools/lockcheck.py"), str(d / "ref_plate.mp4"), str(d / "ref_normal.mp4"),
                           str(d / "ref_mask.mp4"), str(d / "lc_ceiling"), "--tol", str(R["tol"])],
                          check=True, capture_output=True, text=True).stdout.strip()
    with open(d / "scores.txt", "a") as fh:
        fh.write(f"final {n} frames: {lc}\nceiling (Blender plate): {ceil}\n")
    print(lc); print("ceiling:", ceil)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    lab = lambda t: (f"scale=640:352,drawbox=x=0:y=0:w=iw:h=30:color=black@0.55:t=fill,"
                     f"drawtext=fontfile={font}:text='{t}':x=10:y=7:fontsize=17:fontcolor=white")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "ref_plate.mp4"), "-i", str(out), "-i", str(d / "lc_final_overlay.mp4"),
                    "-filter_complex", f"[0:v]{lab('Blender blockout (our rig)')}[a];[1:v]{lab('H3 render (skeleton look)')}[b];"
                                       f"[2:v]{lab('Lock check, our edges in red over H3')}[c];[a][b][c]hstack=3[v]",
                    "-map", "[v]", "-map", "1:a", "-frames:v", str(n), "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-c:a", "copy", str(d / f"{run}_triptych.mp4")], check=True)
    print("final", out, d / f"{run}_triptych.mp4")


def inside(run):
    """The shot's stock hand (Blender) | its control (rig joints + keys) | H3, take frames, with the take's sound."""
    R = RUNS[run]; d = ROOT / run; K, n = R["densify"], R["keep"]
    nc = len(R["clips"])
    ctl = R.get("control_kind", "ncanny")
    decimate(_cat([COMFY_IN / cin(src(run), k, ctl) for k in range(nc)], d / f"all_{ctl}.mp4", R.get("overlap", 0)),
             d / "ref_control.mp4", K, n)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    lab = lambda t: (f"scale=640:352,drawbox=x=0:y=0:w=iw:h=30:color=black@0.55:t=fill,"
                     f"drawtext=fontfile={font}:text='{t}':x=10:y=7:fontsize=17:fontcolor=white")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "ref_plate.mp4"), "-i", str(d / "ref_control.mp4"),
                    "-i", str(d / f"{run}_final.mp4"),
                    "-filter_complex", f"[0:v]{lab('our rig, the stock MPFB2 hand (Blender)')}[a];"
                                       f"[1:v]{lab('control, rig joints + keys')}[b];[2:v]{lab('H3 render')}[c];"
                                       "[a][b][c]hstack=3[v]",
                    "-map", "[v]", "-map", "2:a", "-frames:v", str(n), "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-c:a", "copy", str(d / f"{run}_inside.mp4")], check=True)
    print("inside", d / f"{run}_inside.mp4")


def final(run):
    if "keep" in RUNS[run]:
        return final_shot(run)
    R = RUNS[run]; d = ROOT / run; K = R["densify"]; ref = RUNS[R["ref"]]; rd = ROOT / R["ref"]
    done = []
    for k in range(len(R["clips"])):
        if not out_mp4(run, k).exists():
            break
        done.append(out_mp4(run, k))
    n = sum(R["clips"][:len(done)]) // K                 # the take's own frames covered by the rendered clips
    lst = d / "final_clips.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in done))
    t0 = ref["f0"] / 24.0
    out = d / f"{run}_final.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-ss", f"{t0:.4f}",
                    "-t", f"{n / 24.0:.4f}", "-i", str(ref["sound"]), "-map", "0:v", "-map", "1:a",
                    "-vf", f"select='not(mod(n\\,{K}))',setpts=N/24/TB", "-r", "24", "-frames:v", str(n),
                    "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(out)],
                   check=True)
    lc = subprocess.run([PY, str(REPO / "h3_tools/lockcheck.py"), str(out), str(rd / "c0_normal.mp4"), str(rd / "c0_mask.mp4"),
                         str(d / "lc_final"), "--tol", str(R["tol"])], check=True, capture_output=True, text=True).stdout.strip()
    with open(d / "scores.txt", "a") as fh:
        fh.write(f"final ({len(done)} clips, {n} frames): {lc}\n")
    print(lc)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    lab = lambda t: (f"scale=640:352,drawbox=x=0:y=0:w=iw:h=30:color=black@0.55:t=fill,"
                     f"drawtext=fontfile={font}:text='{t}':x=10:y=7:fontsize=17:fontcolor=white")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-start_number", str(ref["f0"]),
                    "-i", str(rd / "plate/f_%05d.png"), "-i", str(out), "-i", str(d / "lc_final_overlay.mp4"),
                    "-filter_complex", f"[0:v]{lab('Blender blockout (our rig)')}[a];[1:v]{lab('H3 render (skeleton look)')}[b];"
                                       f"[2:v]{lab('Lock check, our edges in red over H3')}[c];[a][b][c]hstack=3[v]",
                    "-map", "[v]", "-map", "1:a", "-frames:v", str(n), "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-c:a", "copy", str(d / f"{run}_triptych.mp4")], check=True)
    print("final", out, d / f"{run}_triptych.mp4")


if __name__ == "__main__":
    cmd, run = sys.argv[1:3]
    if cmd in ("passes", "all"):
        if "keep" in RUNS[run]:
            shot_edit(run)
        passes(run)
    if cmd in ("run", "all"):
        run_all(run, upto=int(sys.argv[3]) if len(sys.argv) > 3 else None)
    if cmd == "final":
        final(run)
    if cmd == "inside":
        inside(run)
    if cmd == "controls":
        controls(run, sys.argv[3])
    if cmd == "adopt":
        adopt(run, sys.argv[3], int(sys.argv[4]))
    if cmd == "nohands":
        nohands(run)
    if cmd == "overlay":
        overlay_inputs(run)
    if cmd == "score":
        for k in range(len(RUNS[run]["clips"])):
            score(run, k)
    if cmd in ("join", "all"):
        join(run)
