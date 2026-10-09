"""Look at every frame. Renders per-hand tracking close-ups for every frame of a piano anim JSON, tiles them into
contact sheets, computes per-frame numeric flags, and collects visual reviews.

    python -m blockout.review all  ANIM.json OUTDIR [--jobs 20]      # render + meshqa + flags + sheets
    python -m blockout.review render|meshqa|flags|sheets ANIM.json OUTDIR
    python -m blockout.review collect OUTDIR                          # merge OUTDIR/reviews/*.json -> visual_review.json
    python -m blockout.review compare OUTDIR                          # numeric flags vs visual review -> compare.json

OUTDIR layout: tiles/{R,L}_{angle}/f_#####.png, sheets/{R,L}_{angle}/s_###.png, meshqa.json, review.json,
reviews/*.json (one per reviewer), visual_review.json.
Reviewer instructions: blockout/review_prompt.md.
"""
import argparse, json, math, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BLENDER_PY = os.path.join(HERE, "blender_piano.py")
ANGLES = {"front3q": "body", "side": "body", "front": "body,case"}      # angle -> --hide
HANDS = ("R", "L")
TILE = (320, 240)
COLS, ROWS = 6, 4
PER_SHEET = COLS * ROWS


def _run(cmd):
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, cwd=ROOT)


def _chunks(n, k):
    step = math.ceil(n / k)
    return [(a, min(a + step, n)) for a in range(0, n, step)]


def render(anim, out, jobs=20):
    n = len(json.load(open(anim))["frames"])
    work = []
    per = max(1, jobs // (len(HANDS) * len(ANGLES)))
    for h in HANDS:
        for ang, hide in ANGLES.items():
            d = os.path.join(out, "tiles", f"{h}_{ang}")
            for a, b in _chunks(n, per):
                work.append(["blender", "-b", "--factory-startup", "-P", BLENDER_PY, "--", anim, d, "--view", "handcam",
                             "--hand", h, "--angle", ang, "--hide", hide, "--res", f"{TILE[0]}x{TILE[1]}",
                             "--lens", "45", "--frames", f"{a}:{b}"])
    with ThreadPoolExecutor(jobs) as ex:
        for r in ex.map(_run, work):
            if r.returncode:
                print(r.stderr[-500:], file=sys.stderr)


def meshqa(anim, out, jobs=20):
    n = len(json.load(open(anim))["frames"])
    parts = []
    work = []
    for k, (a, b) in enumerate(_chunks(n, jobs)):
        f = os.path.join(out, f"meshqa_{k}.json")
        parts.append(f)
        work.append(["blender", "-b", "--factory-startup", "-P", BLENDER_PY, "--", anim, out, "--meshqa", f,
                     "--frames", f"{a}:{b}"])
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(_run, work))
    mq = {"rest_pairs": None, "frames": {}}
    for f in parts:
        if os.path.exists(f):
            d = json.load(open(f)); mq["rest_pairs"] = d["rest_pairs"]; mq["frames"].update(d["frames"]); os.remove(f)
    json.dump(mq, open(os.path.join(out, "meshqa.json"), "w"))
    print("meshqa frames with intersections:", len(mq["frames"]))


# ------------------------------------------------------------------ numeric flags
def flags(anim_path, out, excess_deg=5.0, pop_m=0.06, tip_off_m=0.015):
    sys.path.insert(0, ROOT)
    from blockout import piano
    A = json.load(open(anim_path))
    fr = A["frames"]
    mq = json.load(open(os.path.join(out, "meshqa.json")))["frames"] if os.path.exists(os.path.join(out, "meshqa.json")) else {}
    kb = {k["pitch"]: k for k in A["keyboard"]}
    res = []
    prev = None
    for i, f in enumerate(fr):
        F = {}
        # 1. joint angles outside ANAT (fingers 2-5) by more than excess_deg
        for h, hd in f["hands"].items():
            bad = []
            for fi, (m, pi, di, ab, l1, l2) in enumerate(piano.finger_angles(hd["wrist"], hd["fingers"], h), 2):
                for k, v, (lo, hi) in (("mcp", m, piano.ANAT["mcp"]), ("pip", pi, piano.ANAT["pip"]), ("dip", di, piano.ANAT["dip"])):
                    e = max(lo - v, v - hi, 0.0)
                    if e > excess_deg:
                        bad.append([f"{h}{fi}", k, round(v, 1), round(e, 1)])
                for k, v, lim in (("abd", ab, piano.ANAT["abd"]), ("lat", max(abs(l1), abs(l2)), piano.ANAT["lat"])):
                    e = max(abs(v) - lim, 0.0)
                    if e > excess_deg:
                        bad.append([f"{h}{fi}", k, round(v, 1), round(e, 1)])
            if bad:
                F.setdefault("anatomy", {})[h] = bad
        # 2. skinned mesh self-intersection
        if str(i) in mq:
            F["mesh"] = mq[str(i)]
        # 3. hand-to-hand overlap
        if "R" in f["hands"] and "L" in f["hands"]:
            A_, B_ = piano._capsules(f["hands"]["L"]), piano._capsules(f["hands"]["R"])
            d = 0.0
            for a0, a1, ra in A_:
                for b0, b1, rb in B_:
                    if min(np.linalg.norm(a0 - b0), np.linalg.norm(a1 - b1)) > 0.25:
                        continue
                    d = max(d, ra + rb - piano._seg_dist(a0, a1, b0, b1))
            if d > 0.003:
                F["hands_overlap_mm"] = round(d * 1000, 1)
        # 4. fingertip off its key / pressed key with no finger
        off = []
        for p, who in f.get("press", {}).items():
            if f["keys"].get(p, 0) < 0.3:
                continue
            h, fi = who[0], int(who[1]) - 1
            tip = np.asarray(f["hands"][h]["fingers"][fi][3], float)
            c = piano.contact(int(p), thumb=fi == 0)
            dx = abs(tip[0] - c[0])
            if dx > tip_off_m:
                off.append([who, int(p), round(dx * 1000, 1)])
        if off:
            F["tip_off_key"] = off
        held = [p for p, d_ in f["keys"].items() if d_ > 0.3 and p not in f.get("press", {})]
        if held:
            F["key_no_finger"] = held
        # 5. pops: any joint moving more than pop_m in one frame
        if prev is not None:
            for h, hd in f["hands"].items():
                pts = np.array([hd["wrist"]] + [c for ch in hd["fingers"] for c in ch], float)
                pp = np.array([prev[h]["wrist"]] + [c for ch in prev[h]["fingers"] for c in ch], float)
                mv = np.linalg.norm(pts - pp, axis=1).max()
                if mv > pop_m:
                    F.setdefault("pop_m", {})[h] = round(float(mv), 3)
        prev = f["hands"]
        res.append(F)
    json.dump({"thresholds": {"anat_excess_deg": excess_deg, "pop_m": pop_m, "tip_off_m": tip_off_m},
               "frames": {str(i): F for i, F in enumerate(res) if F}, "n_frames": len(res)},
              open(os.path.join(out, "review.json"), "w"))
    c = {}
    for F in res:
        for k in F:
            c[k] = c.get(k, 0) + 1
    print("frames flagged by kind:", c, "any:", sum(1 for F in res if F), "of", len(res))


# ------------------------------------------------------------------ sheets
def sheets(anim, out):
    from PIL import Image, ImageDraw
    n = len(json.load(open(anim))["frames"])
    for h in HANDS:
        for ang in ANGLES:
            tdir = os.path.join(out, "tiles", f"{h}_{ang}")
            sdir = os.path.join(out, "sheets", f"{h}_{ang}")
            os.makedirs(sdir, exist_ok=True)
            for s, a in enumerate(range(0, n, PER_SHEET)):
                im = Image.new("RGB", (COLS * TILE[0], ROWS * TILE[1]))
                dr = ImageDraw.Draw(im)
                for k, i in enumerate(range(a, min(a + PER_SHEET, n))):
                    p = os.path.join(tdir, f"f_{i:05d}.png")
                    x, y = (k % COLS) * TILE[0], (k // COLS) * TILE[1]
                    if os.path.exists(p):
                        im.paste(Image.open(p).convert("RGB"), (x, y))
                    dr.rectangle([x, y, x + 62, y + 16], fill=(0, 0, 0))
                    dr.text((x + 3, y + 2), f"#{i:03d} {h}", fill=(255, 255, 0))
                    dr.rectangle([x, y, x + TILE[0] - 1, y + TILE[1] - 1], outline=(70, 70, 120))
                im.save(os.path.join(sdir, f"s_{s:03d}.png"))
    print("sheets written to", os.path.join(out, "sheets"))


def collect(out):
    items = []
    for f in sorted(os.listdir(os.path.join(out, "reviews"))):
        try:
            d = json.load(open(os.path.join(out, "reviews", f)))
            items += d if isinstance(d, list) else d.get("defects", [])
        except Exception as e:
            print("bad review file", f, e)
    json.dump(items, open(os.path.join(out, "visual_review.json"), "w"), indent=0)
    print(len(items), "visual defects from", len(os.listdir(os.path.join(out, "reviews"))), "reviewer files")


def compare(out, sev_min=2):
    rv = json.load(open(os.path.join(out, "review.json")))
    n = rv["n_frames"]
    auto = rv["frames"]
    vis = json.load(open(os.path.join(out, "visual_review.json")))
    vis = [v for v in vis if int(v.get("severity", 1)) >= sev_min]
    vf = {}
    for v in vis:
        vf.setdefault(int(v["frame"]), []).append(v)
    # "hard" numeric flags: those that describe a visible break (anatomy>15 deg, mesh, overlap>3mm, pop)
    def hard(F):
        k = []
        if "anatomy" in F and any(b[3] > 15 for bl in F["anatomy"].values() for b in bl): k.append("anatomy15")
        if "mesh" in F: k.append("mesh")
        if "hands_overlap_mm" in F: k.append("hands_overlap")
        if "pop_m" in F: k.append("pop")
        if "tip_off_key" in F: k.append("tip_off_key")
        if "key_no_finger" in F: k.append("key_no_finger")
        return k
    H = {int(i): hard(F) for i, F in auto.items() if hard(F)}
    both = sorted(set(H) & set(vf)); only_n = sorted(set(H) - set(vf)); only_v = sorted(set(vf) - set(H))
    by_def = {}
    for v in vis:
        by_def[v["defect"]] = by_def.get(v["defect"], 0) + 1
    cmp = {"frames": n, "visual_sev>=": sev_min, "visual_frames": len(vf), "numeric_hard_frames": len(H),
           "both": len(both), "only_numeric": len(only_n), "only_visual": len(only_v),
           "visual_defect_counts": by_def, "only_visual_frames": only_v,
           "agreement_by_numeric_kind": {k: {"flagged": sum(1 for i, ks in H.items() if k in ks),
                                             "also_seen": sum(1 for i, ks in H.items() if k in ks and i in vf)}
                                         for k in ("anatomy15", "mesh", "hands_overlap", "pop", "tip_off_key", "key_no_finger")}}
    json.dump(cmp, open(os.path.join(out, "compare.json"), "w"), indent=1)
    print(json.dumps(cmp, indent=1)[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("a"); ap.add_argument("b", nargs="?")
    ap.add_argument("--jobs", type=int, default=20)
    x = ap.parse_args()
    if x.cmd in ("collect", "compare"):
        {"collect": collect, "compare": compare}[x.cmd](x.a)
        sys.exit()
    anim, out = x.a, x.b
    os.makedirs(out, exist_ok=True)
    t = time.time()
    steps = {"render": lambda: render(anim, out, x.jobs), "meshqa": lambda: meshqa(anim, out, x.jobs),
             "flags": lambda: flags(anim, out), "sheets": lambda: sheets(anim, out)}
    for s in (steps if x.cmd == "all" else [x.cmd]):
        t0 = time.time(); steps[s](); print(f"[{s}] {time.time() - t0:.1f}s")
    print(f"total {time.time() - t:.1f}s")
