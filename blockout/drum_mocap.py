"""Drummer motion from mocap, in beat time: a real drummer's head and trunk motion, for drums.py --mocap.

Build once per drummer (needs librosa and ffmpeg; the drummer itself only reads the JSON):

    python -m blockout.drum_mocap --build blockout/drum_body_sources.json blockout/drum_mocap_chad.json
    python -m blockout.drum_mocap --build blockout/drum_body_sources_carey.json blockout/drum_mocap_carey.json
                                  --ref blockout/drum_body_ref_carey.json --source all

Reads sources of a drum_body_sources.json (the first, one by id, a comma list or "all": every clip of one drummer
pools into one style): each source's SAM 3D Body BVH (h3band/sam3d_mocap.py) and its clip's audio.
For each camera shot:
  beats   librosa's beat tracker on the clip's audio. A shot is used when its beats are steady (inter-beat
          interval CV under IBI_CV_MAX) and the drummer's strokes (his wrists' low points) lock onto the audio's
          8th-note grid (Rayleigh test, p under SYNC_P_MAX), or failing that its 16ths or triplets at
          SYNC_P_MAX over the number of grids (SYNC_GRIDS: a 16th-note player, Carey, or a triplet player, Baker,
          splits their strokes evenly over the 8ths): then the audio is what they play. On the 8th grid the strokes'
          mean offset is taken out of the shot's timing (a multi-camera edit syncs each camera a frame or two
          apart). On the others it is where they play in the beat as much as the camera's sync, so it is left: those
          clips are one camera each, picture and sound together
  motion  per frame, as performer.rot angles (pitch down, yaw toward the drummer's left, roll toward his right):
          head    the head joint's frame relative to the trunk (up = pelvis -> neck, right = the shoulder line), minus
                  the shot median, so our drummer keeps his own gaze at his kit and takes on the motion. The roll
                  channel is the head's tilt against the trunk as drum_body_score sees it (tilt()), not the Euler
                  roll: with the head pitched down on the trunk, a turn shows as a tilt too, the more the further
                  down, and our drummer looks further down at his kit than Carey or Baker do. tile() turns the tilt
                  back into a roll at our head's own pitch
          trunk   the trunk frame relative to its mean over the shot (the camera holds still within a shot)
          low-passed at LOWPASS_HZ inside the shot, then resampled to RATE samples per beat from the shot's first
          whole beat
Bar phase is not known (the audio is the full band; the snare does not stand out of it), so the segments keep the
clip's own beat count: every segment's "beat0" is a beat index into the one continuous track, and drums.py puts
even clip beats on the song's downbeat parity.
  gain    shots that pass are not a fair sample of the drummer's playing (Chad's are the clip's busiest), so each channel
          gets a gain that brings the used shots' motion to the whole drum_body_score reference's, with room left
          for the motion our drummer makes himself (gains(): nod, head turn and tilt; trunk sway and shoulder turn)

In the drummer (tile): seeded random chunks of CHUNKS beats cut from the segments are laid end to end on the song's
beat grid, each crossfaded into the next over one beat; another seed gives another order.
"""
import argparse, json, math, os, subprocess

import numpy as np

from . import drum_body_score as D

RATE = 24                   # samples per beat
LOWPASS_HZ = 6.0            # the mocap's frame-to-frame jitter goes; 8th-note motion (~4 Hz at 120 bpm) stays
IBI_CV_MAX = 0.12
SYNC_P_MAX = 0.05           # Rayleigh test: the strokes must lock to the audio's 8th-note grid ...
SYNC_GRIDS = (2, 4, 3)      # ... or, at SYNC_P_MAX over the number of grids, its 16ths or triplets
GRID_NAMES = {2: "8ths", 4: "16ths", 3: "triplets"}
SR, HOP = 22050, 256
CHUNKS = (4, 6, 8)          # beats; even, so a chunk ends on the parity it started on
XFADE = 1.0                 # beats
GAIN_MEASURES = {"head_pitch": "nod", "head_yaw": "head_turn", "head_roll": "head_tilt",
                 "trunk_pitch": "trunk_sway", "trunk_yaw": "shoulder_turn", "trunk_roll": "trunk_sway"}
BASE_MOTION = {"nod": 9.4, "head_turn": 14.3, "head_tilt": 6.6,     # deg: our drummer's own under --mocap with the
               "trunk_sway": 6.2, "shoulder_turn": 6.1}            # head's (trunk's) channels off: looks, leans, twists
                                                                    # (technique groove, big kit, groove emotion)
GAIN_KEEP = 0.5             # at least this share of the drummer's own motion stays where ours already fills theirs
GAIN_RANGE = (0.4, 2.0)
CHANNELS = ("head_pitch", "head_yaw", "head_roll", "trunk_pitch", "trunk_yaw", "trunk_roll")


def decompose(M):
    """(..., 3, 3) rotations -> pitch_down, yaw, roll with M = performer.rot(pitch_down, yaw, roll)."""
    pitch = -np.arcsin(np.clip(M[..., 2, 1], -1, 1))
    roll = np.arctan2(-M[..., 2, 0], M[..., 2, 2])
    yaw = np.arctan2(-M[..., 0, 1], M[..., 1, 1])
    return pitch, yaw, roll


def frames(pts, hf, hu):
    """Trunk and head frames (F, 3, 3), columns = right, forward, up (performer / anim axes)."""
    up = D._n(pts["neck"] - pts["pelvis"])
    right = D._orth(pts["shR"] - pts["shL"], up)
    T = np.stack([right, np.cross(up, right), up], -1)
    f = D._n(hf)
    r = D._n(np.cross(f, hu))
    H = np.stack([r, f, np.cross(r, f)], -1)
    return T, H


def _mean_frame(T):
    up = D._n(T[:, :, 2].mean(0))
    right = D._orth(T[:, :, 0].mean(0)[None], up[None])[0]
    return np.stack([right, np.cross(up, right), up], -1)


def audio(path):
    """Mono float32 at SR, decoded by ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def track(y):
    """Beat times (s)."""
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    _, bt = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP, units="time", tightness=400)
    return np.asarray(bt, float)


def _lows(z, t, prom):
    """Times of a wrist's low points (height track z at frame times t), sub-frame by a parabola through each."""
    from scipy.signal import find_peaks
    idx, _ = find_peaks(-z, prominence=prom)
    out = []
    for i in idx[(idx > 0) & (idx < len(z) - 1)]:
        a, b, c = z[i - 1], z[i], z[i + 1]
        den = a - 2 * b + c
        out.append(t[i] + (0.5 * (a - c) / den if den else 0.0) * (t[1] - t[0]))
    return np.asarray(out, float)


def _sync(lows, beats, sub=2):
    """Stroke low points against the audio's grid of `sub` notes per beat (2: 8ths): (mean offset s, + = the
    picture's strokes late; resultant length R; Rayleigh p). Strokes the audio does not hold spread over the grid:
    R near 0, p near 1."""
    if len(lows) < 3:
        return 0.0, 0.0, 1.0
    ph = 2 * np.pi * ((sub * beat_pos(lows, beats)) % 1.0)
    z = np.mean(np.exp(1j * ph))
    n, R = len(ph), float(abs(z))
    ibi = float(np.median(np.diff(beats)))
    return float(np.angle(z) / (2 * np.pi) * ibi / sub), R, float(math.exp(-n * R * R))


def _lowpass(x, fps):
    from scipy.signal import butter, filtfilt
    b, a = butter(2, LOWPASS_HZ, fs=fps)
    return filtfilt(b, a, x, axis=0) if len(x) > 3 * max(len(a), len(b)) else x


def gains(ref, used):
    """Per channel, the gain that brings the used shots' motion ("source:shot" keys) to the whole reference's,
    with room left for our drummer's own: the drum_body_score reference's pooled median of the channel's measure
    (GAIN_MEASURES), less BASE_MOTION in quadrature (two unrelated motions' ranges add about so) but no less than
    GAIN_KEEP of it, over the window-weighted mean of the used shots' medians. Shots that pass the sync check are
    not a fair sample of a drummer's playing (Chad's are the clip's busiest). Measured against the drummer's own
    reference, from the same sources."""
    out = {}
    for c, k in GAIN_MEASURES.items():
        spec = ref["measures"][k]
        ps = [spec["per_shot"][key] for key in used if key in spec["per_shot"]]
        mine = sum(b["p50"] * b["n"] for b in ps) / max(sum(b["n"] for b in ps), 1)
        want = spec["pooled"]["p50"]
        want = math.sqrt(max(want ** 2 - BASE_MOTION[k] ** 2, (GAIN_KEEP * want) ** 2))
        out[c] = round(float(np.clip(want / max(mine, 1e-6), *GAIN_RANGE)), 3)
    return out


def tilt(T, H):
    """(F,) radians: the head's tilt against the trunk as drum_body_score measures it (its head_roll), the head's
    up axis toward the trunk's right, from trunk and head frames (frames())."""
    hu = H[:, :, 2]
    return np.arctan2(np.einsum("ij,ij->i", hu, T[:, :, 0]), np.einsum("ij,ij->i", hu, T[:, :, 2]))


def _source(root, s, multi):
    """(segments, shots, beats, mocap report) of one source; shot keys are "source:shot" when sources pool."""
    pts, hf, hu, fps, rep = D.mhr_points(os.path.join(root, s["bvh"]))
    T, H = frames(pts, hf, hu)
    up = T[:, :, 2]
    t = np.arange(len(T)) / fps
    beats = track(audio(os.path.join(root, s["video"])))
    bidx = np.arange(len(beats), dtype=float)
    arm = float(np.median(np.linalg.norm(pts["elR"] - pts["shR"], axis=1) + np.linalg.norm(pts["wrR"] - pts["elR"], axis=1)))
    segs, shots = [], {}
    for sh in s["shots"]:
        lo, hi = sh["t"][0] + D.TRIM, sh["t"][1] - D.TRIM
        rec = {"view": sh.get("view"), "t": [round(lo, 3), round(hi, 3)]}
        shots[f"{s['id']}:{sh['id']}" if multi else sh["id"]] = rec
        inb = beats[(beats >= lo) & (beats <= hi)]
        if len(inb) < 4:
            rec["skipped"] = f"{len(inb)} beats tracked in the audio"
            continue
        ibi = np.diff(inb)
        rec["bpm"] = round(60.0 / float(np.median(ibi)), 1)
        rec["ibi_cv"] = round(float(np.std(ibi) / np.mean(ibi)), 3)
        sl = (t >= lo) & (t <= hi)
        lows = np.concatenate([_lows(D._dot(pts["wr" + side] - pts["sh" + side], up)[sl], t[sl], 0.02 * arm)
                               for side in "RL"])
        off, R, p = _sync(lows, beats)
        if p > SYNC_P_MAX:      # 16ths or triplets split evenly over the 8th grid: try the other grids
            sub, (_, R2, p2) = min(((k, _sync(lows, beats, k)) for k in SYNC_GRIDS[1:]), key=lambda e: e[1][2])
            if p2 * len(SYNC_GRIDS) <= SYNC_P_MAX:
                off, R, p = 0.0, R2, p2 * len(SYNC_GRIDS)
                rec["sync_grid"] = GRID_NAMES[sub]
        rec.update(wrist_lows=len(lows), sync_R=round(R, 2), sync_p=round(p, 4), sync_offset_ms=round(1000 * off, 1))
        if rec["ibi_cv"] > IBI_CV_MAX:
            rec["skipped"] = f"beats not steady (IBI CV {rec['ibi_cv']})"
            continue
        if p > SYNC_P_MAX:
            rec["skipped"] = f"strokes not locked to the audio's 8th-note grid (Rayleigh p {p:.3f})"
            continue
        # motion, per frame inside the shot
        Ts, Hs = T[sl], H[sl]
        hp, hy, _ = decompose(np.swapaxes(Ts, 1, 2) @ Hs)
        hr = tilt(Ts, Hs)
        tp, ty, tr = decompose(np.swapaxes(_mean_frame(Ts), 0, 1)[None] @ Ts)
        X = np.stack([hp, hy, hr, tp, ty, tr], 1)
        X -= np.median(X, 0)
        X = _lowpass(X, fps)
        # to beat time, from the first whole beat inside the shot to the last sample inside it
        # (the picture runs `off` ahead of the sound in this shot: audio time ta shows at video time ta + off)
        b_lo, b_hi = np.interp([lo - off, hi - off], beats, bidx)
        b0 = int(math.ceil(b_lo))
        bs = b0 + np.arange(int((b_hi - b0) * RATE) + 1) / RATE
        ts = np.interp(bs, bidx, beats)
        data = np.stack([np.interp(ts + off, t[sl], X[:, c]) for c in range(len(CHANNELS))], 1)
        rec["beats"] = round((len(data) - 1) / RATE, 2)
        seg = {"shot": sh["id"], "view": sh.get("view"), "beat0": b0, "t0": round(float(ts[0]), 3),
               "data": np.round(data, 4).tolist()}
        segs.append(dict({"source": s["id"]}, **seg) if multi else seg)
    return segs, shots, beats, rep


def build(sources_json, out_json, source_id=None, ref_path=D.REF):
    src = json.load(open(sources_json))
    root = os.path.dirname(os.path.abspath(sources_json))
    if source_id == "all":
        use = src["sources"]
    elif source_id and "," in source_id:
        ids = source_id.split(",")
        use = [x for x in src["sources"] if x["id"] in ids]
    else:
        use = [next(x for x in src["sources"] if source_id in (None, x["id"]))]
    multi = len(use) > 1
    segs, shots, beats, reps = [], {}, {}, {}
    for s in use:
        g, sh, b, rep = _source(root, s, multi)
        segs += g
        shots.update(sh)
        beats[s["id"]] = [round(float(x), 3) for x in b]
        reps[s["id"]] = rep
    if not segs:
        raise SystemExit("no shot passed the beat and sync checks")
    alld = np.concatenate([np.asarray(g["data"]) for g in segs])
    gain = gains(json.load(open(ref_path)), [f"{g.get('source', use[0]['id'])}:{g['shot']}" for g in segs])
    made = f"python -m blockout.drum_mocap --build {os.path.basename(sources_json)} {os.path.basename(out_json)}"
    if ref_path != D.REF:
        made += f" --ref {os.path.basename(ref_path)}"
    if multi:
        made += f" --source {source_id}"
    head = ({"source": src.get("drummer", use[0]["id"]), "drummer": src.get("drummer"), "sources": [x["id"] for x in use],
             "content": [x.get("content") for x in use]} if multi
            else {"source": use[0]["id"], "content": use[0].get("content")})
    out = {"version": 1, **head, "made_with": made, "rate": RATE, "channels": CHANNELS,
           "units": "radians, performer.rot convention: pitch down, yaw toward the drummer's left, roll toward his right",
           "head": "head relative to the trunk, minus the shot median; head_roll is the tilt against the trunk "
                   "(drum_body_score's head_roll), turned into a roll at the player's head pitch",
           "trunk": "trunk relative to its shot mean",
           "lowpass_hz": LOWPASS_HZ, "gain": gain, "gain_from": f"{os.path.basename(ref_path)}: "
           + "; ".join(f"{c}: {k}" for c, k in GAIN_MEASURES.items())
           + f"; less our drummer's own (deg) {BASE_MOTION} in quadrature", "mocap": reps if multi else reps[use[0]["id"]],
           "audio_beats": beats if multi else beats[use[0]["id"]], "shots": shots,
           "stats_deg": {c: [round(float(v), 1) for v in np.degrees(np.percentile(alld[:, i], (5, 50, 95)))]
                         for i, c in enumerate(CHANNELS)},
           "segments": segs}
    json.dump(out, open(out_json, "w"), separators=(",", ":"))
    print(f"wrote {out_json}: {len(segs)} segments, {sum(len(g['data']) - 1 for g in segs) / RATE:.1f} beats")
    for k, v in shots.items():
        print(f"  {k}: {json.dumps(v)}")
    print("  p5/p50/p95 deg: " + "  ".join(f"{c} {v}" for c, v in out["stats_deg"].items()))
    print(f"  gain: {gain}")
    return out


# ---------------------------------------------------------------- in the drummer

def load(path):
    st = json.load(open(path))
    for g in st["segments"]:
        g["data"] = np.asarray(g["data"], float)
    return st


def beat_pos(times, beats):
    """Fractional beat index at each time; linear past the ends at the end intervals."""
    beats = np.asarray(beats, float)
    i = np.arange(len(beats), dtype=float)
    b = np.interp(times, beats, i)
    if len(beats) > 1:
        b = np.where(times < beats[0], (times - beats[0]) / (beats[1] - beats[0]), b)
        b = np.where(times > beats[-1], i[-1] + (times - beats[-1]) / (beats[-1] - beats[-2]), b)
    return b


def beat_time(k, beats):
    """Time of fractional beat index k (beat_pos's inverse)."""
    beats = np.asarray(beats, float)
    if k < 0:
        return float(beats[0] + k * (beats[1] - beats[0]))
    if k > len(beats) - 1:
        return float(beats[-1] + (k - len(beats) + 1) * (beats[-1] - beats[-2]))
    return float(np.interp(k, np.arange(len(beats)), beats))


def _sample(g, x):
    """Segment g's channels at clip beat offset x (beats from its first sample)."""
    d = g["data"]
    k = np.clip(np.asarray(x, float) * RATE, 0, len(d) - 1)
    i0 = np.minimum(np.floor(k).astype(int), len(d) - 2)
    u = (k - i0)[..., None]
    return d[i0] * (1 - u) + d[i0 + 1] * u


def plan(style, b_lo, b_hi, seed=0, parity=0):
    """Chunks [(song_beat, beats, segment index, offset in beats)] covering song beats b_lo..b_hi. Chunks start on
    song beats of `parity` and on even clip beats; each segment holds its chunk plus the beat it fades out over."""
    rng = np.random.default_rng(seed + 7919)
    opts = []
    for si, g in enumerate(style["segments"]):
        n = (len(g["data"]) - 1) / RATE
        for L in CHUNKS:
            opts += [(L, si, off) for off in range(int(g["beat0"]) % 2, math.floor(n - L - XFADE) + 1, 2)]
    if not opts:
        raise ValueError("mocap style has no segment long enough for a chunk")
    k = int(math.floor(b_lo)) - 2
    k -= (k - parity) % 2
    out, last = [], None
    while k <= b_hi + 1:
        pick = [o for o in opts if o[1:] != last] or opts
        L, si, off = pick[int(rng.integers(len(pick)))]
        out.append((k, L, si, off))
        last = (si, off)
        k += L
    return out


def tile(style, beats, times, seed=0, parity=0, pitch=None):
    """(N, 6) mocap channels (CHANNELS) at `times`, laid on the song's `beats`, times the style's gains; and the
    chunk plan. pitch (N,): the player's head pitch on his trunk, radians, where the tilt channel turns into the
    roll that shows that tilt (a turn of a head pitched down tilts it too: tilt())."""
    b = beat_pos(np.asarray(times, float), beats)
    chunks = plan(style, float(b.min()), float(b.max()), seed, parity)
    segs = style["segments"]
    out = np.zeros((len(b), len(CHANNELS)))
    starts = np.array([c[0] for c in chunks], float)
    ci = np.clip(np.searchsorted(starts, b, side="right") - 1, 0, len(chunks) - 1)
    for c, (k, L, si, off) in enumerate(chunks):
        m = ci == c
        if not m.any():
            continue
        x = b[m] - k
        v = _sample(segs[si], off + x)
        if c > 0:       # the chunk before plays on into this one's first beat, fading out
            pk, _, psi, poff = chunks[c - 1]
            u = np.clip(x / XFADE, 0, 1)[:, None]
            u = u * u * (3 - 2 * u)
            v = (1 - u) * _sample(segs[psi], poff + (b[m] - pk)) + u * v
        out[m] = v
    g = style.get("gain", {})     # per channel, or (older styles) per part: "head", "trunk"
    out *= np.array([g.get(c, g.get(c.split("_")[0], 1.0)) for c in CHANNELS])
    if pitch is not None:
        p = np.asarray(pitch, float)
        out[:, 2] = out[:, 2] * np.cos(p) + out[:, 1] * np.sin(p)
    return out, chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", nargs=2, metavar=("SOURCES_JSON", "OUT_JSON"), required=True)
    ap.add_argument("--source", help="source id, a comma list of them, or all (default: the first); "
                                     "more than one pools into one style")
    ap.add_argument("--ref", default=D.REF, help="drum_body_score reference, for the head and trunk gains")
    a = ap.parse_args()
    build(*a.build, a.source, a.ref)


if __name__ == "__main__":
    main()
