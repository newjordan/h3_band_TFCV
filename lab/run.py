"""Piano-hand lab: run one experiment over the whole Knight's Suite on toymaker + sparky + turbo, record it.

    python lab/run.py EXP_ID            # EXP_ID defined in lab/experiments.json
    python lab/run.py --plots           # regenerate lab/plots/*.png from lab/results.csv

An experiment = {id, leg, hypothesis, change, ovr (PIANO_OVR json), env ("K=V ..."), parent}. The suite (Clair de
Lune -> Rachmaninoff -> Chopin, 39.9-390.5 s) is cut into 10 s windows, each solved independently on whichever host
slot is free; blockout/eval_suite.py scores every note. Results: lab/runs/EXP_ID/ (per-window JSON, summary.json),
one row appended to lab/results.csv. Metrics: key sync (whole suite and per section), miss kinds, wrist deviation
against the forearm (median, p95), wrist shake (high-pass rms, mm), finger tremor (joint reversals per joint-second),
joints out of anatomical range per minute.
"""
import csv, json, os, queue, shlex, subprocess, sys, threading, time

LAB = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(LAB)
SUITE = os.path.expanduser("~/h3b_suite/examples/piano")
HOSTS = [("local", 12, os.path.expanduser("~/h3_band_TFCV/.venv_blockout/bin/python"), CODE),
         ("sparky", 18, "/home/frosty40/h3band/venv/bin/python", "/home/frosty40/h3band/code"),
         ("turbo", 28, "/home/frosty40/h3band/venv/bin/python", "/home/frosty40/h3band/code")]
SSH = ["-o", "ControlMaster=auto", "-o", "ControlPath=/tmp/lab-ssh-%r@%h:%p", "-o", "ControlPersist=900"]
WINDOWS = [round(39.9 + 10 * k, 1) for k in range(36) if 39.9 + 10 * k < 390.5]
FIELDS = ["id", "leg", "parent", "change", "started", "minutes", "notes", "ok", "keysync", "debussy", "rach", "chopin",
          "unfingered", "sideways", "short", "shallow", "wrist_dev_median", "wrist_dev_p95", "jitter_mm",
          "tremor_per_joint_s", "anatomy_out_per_min", "ovr", "env"]


def sync(host, code):
    if host == "local":
        return
    subprocess.run(["ssh", *SSH, "-fN", host])          # open the shared connection
    subprocess.run(["rsync", "-az", "-e", "ssh " + " ".join(SSH), "--exclude", ".git", "--exclude", "__pycache__",
                    "--exclude", "lab/runs", CODE + "/", f"{host}:{code}/"], check=True)
    subprocess.run(["rsync", "-az", "-e", "ssh " + " ".join(SSH), f"{SUITE}/knights_suite.mid",
                    f"{SUITE}/knights_suite.feeling.json", f"{host}:{code}/examples/piano/"], check=True)


def run(exp_id):
    E = {e["id"]: e for e in json.load(open(os.path.join(LAB, "experiments.json")))}[exp_id]
    out = os.path.join(LAB, "runs", exp_id)
    os.makedirs(out, exist_ok=True)
    json.dump(E, open(os.path.join(out, "experiment.json"), "w"), indent=1)
    t0 = time.time()
    for h, n, py, code in HOSTS:
        sync(h, code)
    q = queue.Queue()
    for w in WINDOWS:
        q.put(w)

    def worker(h, py, code):
        while True:
            try:
                w = q.get_nowait()
            except queue.Empty:
                return
            mid, fee = (f"{SUITE}/knights_suite.mid", f"{SUITE}/knights_suite.feeling.json") if h == "local" else \
                (f"{code}/examples/piano/knights_suite.mid", f"{code}/examples/piano/knights_suite.feeling.json")
            dst = os.path.join(out, f"w_{w}.json") if h == "local" else f"/tmp/lab_{exp_id}_w_{w}.json"
            cmd = (f"cd {code} && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 {E.get('env', '')} "
                   f"PIANO_OVR={shlex.quote(json.dumps(E.get('ovr') or {}))} {py} -m blockout.eval_suite window "
                   f"{mid} {fee} {w} 10 {dst}")
            if h == "local":
                r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
            else:
                r = subprocess.run(["ssh", *SSH, h, cmd], capture_output=True, text=True)
                if r.returncode == 0:
                    subprocess.run(["scp", "-q", *SSH, f"{h}:{dst}", os.path.join(out, f"w_{w}.json")])
            if r.returncode != 0:
                open(os.path.join(out, f"err_{w}.txt"), "w").write(r.stderr[-4000:])
                q.put(w) if not os.path.exists(os.path.join(out, f"failed_{w}")) else None
                open(os.path.join(out, f"failed_{w}"), "a").write(h + "\n")

    th = [threading.Thread(target=worker, args=(h, py, code)) for h, n, py, code in HOSTS for _ in range(n)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    r = subprocess.run([HOSTS[0][2], "-m", "blockout.eval_suite", "aggregate", out], cwd=CODE, capture_output=True, text=True)
    S = json.load(open(os.path.join(out, "summary.json")))
    row = {"id": exp_id, "leg": E["leg"], "parent": E.get("parent", ""), "change": E["change"],
           "started": time.strftime("%Y-%m-%d %H:%M", time.localtime(t0)), "minutes": round((time.time() - t0) / 60, 1),
           "notes": S["notes"], "ok": S["ok"], "keysync": S["keysync"],
           **{k: float(v.split("= ")[1].rstrip("%")) for k, v in S["sections"].items()},
           **{k: S["miss_kinds"].get(k, 0) for k in ("unfingered", "sideways", "short", "shallow")},
           **{k: S[k] for k in ("wrist_dev_median", "wrist_dev_p95", "jitter_mm", "tremor_per_joint_s", "anatomy_out_per_min")},
           "ovr": json.dumps(E.get("ovr") or {}), "env": E.get("env", "")}
    path = os.path.join(LAB, "results.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, indent=1))
    plots()


def plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(os.path.join(LAB, "results.csv"))))
    if not rows:
        return
    os.makedirs(os.path.join(LAB, "plots"), exist_ok=True)
    legs = sorted({r["leg"] for r in rows})
    col = {l: c for l, c in zip(legs, ["#2b6cb0", "#c05621", "#2f855a", "#805ad5", "#b83280", "#4a5568"])}
    f = lambda r, k: float(r[k]) if r[k] not in ("", "None") else float("nan")
    # 1. key sync over the experiment sequence, per leg (lines + dots)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for l in legs:
        rs = [(i, r) for i, r in enumerate(rows) if r["leg"] == l]
        ax.plot([i for i, _ in rs], [f(r, "keysync") for _, r in rs], "-o", color=col[l], label=l)
        for i, r in rs:
            ax.annotate(r["id"], (i, f(r, "keysync")), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax.axhline(100, color="#999", lw=0.8, ls="--")
    ax.set_xlabel("experiment (run order)"); ax.set_ylabel("key sync, whole suite (%)"); ax.set_title("Key sync by experiment")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(LAB, "plots", "keysync_runs.png"), dpi=140)
    # 2. per-section key sync over runs (lines)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k, c in (("debussy", "#2b6cb0"), ("rach", "#c05621"), ("chopin", "#2f855a")):
        ax.plot(range(len(rows)), [f(r, k) for r in rows], "-o", color=c, label=k, ms=4)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["id"] for r in rows], rotation=60, fontsize=7)
    ax.axhline(100, color="#999", lw=0.8, ls="--"); ax.set_ylabel("key sync (%)"); ax.set_title("Key sync per section")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(os.path.join(LAB, "plots", "keysync_sections.png"), dpi=140)
    # 3-5. trade-offs (scatter, labelled)
    for xk, xl, fn in (("wrist_dev_median", "wrist deviation vs forearm, median (deg)", "tradeoff_wrist.png"),
                       ("jitter_mm", "wrist shake, high-pass rms (mm)", "tradeoff_jitter.png"),
                       ("tremor_per_joint_s", "finger tremor (joint reversals per joint-second)", "tradeoff_tremor.png")):
        fig, ax = plt.subplots(figsize=(7, 5))
        for l in legs:
            rs = [r for r in rows if r["leg"] == l]
            ax.plot([f(r, xk) for r in rs], [f(r, "keysync") for r in rs], "-o", color=col[l], label=l, alpha=0.85)
            for r in rs:
                ax.annotate(r["id"], (f(r, xk), f(r, "keysync")), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.axhline(100, color="#999", lw=0.8, ls="--"); ax.set_xlabel(xl); ax.set_ylabel("key sync (%)")
        ax.set_title("Key sync vs " + xl.split(",")[0]); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(LAB, "plots", fn), dpi=140)
    # 6. miss kinds over runs (lines)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for k, c in (("unfingered", "#4a5568"), ("sideways", "#c05621"), ("short", "#805ad5"), ("shallow", "#2b6cb0")):
        ax.plot(range(len(rows)), [f(r, k) for r in rows], "-o", color=c, label=k, ms=4)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["id"] for r in rows], rotation=60, fontsize=7)
    ax.set_ylabel("missed notes"); ax.set_title("Missed notes by kind"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(LAB, "plots", "misses.png"), dpi=140)
    plt.close("all")


if __name__ == "__main__":
    if sys.argv[1] == "--plots":
        plots()
    else:
        run(sys.argv[1])
