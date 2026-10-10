"""Publish the lab (plots + results table) to the band site: python lab/publish.py"""
import csv, json, os, shutil, subprocess
LAB = os.path.dirname(os.path.abspath(__file__))
W = os.path.expanduser("~/h3/results/band_site/web")
os.makedirs(os.path.join(W, "media", "lab"), exist_ok=True)
for f in os.listdir(os.path.join(LAB, "plots")):
    shutil.copy2(os.path.join(LAB, "plots", f), os.path.join(W, "media", "lab", f))
rows = list(csv.DictReader(open(os.path.join(LAB, "results.csv"))))
E = {e["id"]: e for e in json.load(open(os.path.join(LAB, "experiments.json")))}
table = [["id", "leg", "change", "key sync %", "Debussy", "Rach", "Chopin", "unfingered / sideways / short / shallow",
          "wrist dev med/p95", "shake mm", "tremor"]]
for r in rows:
    table.append([r["id"], r["leg"], r["change"], r["keysync"], r["debussy"], r["rach"], r["chopin"],
                  f'{r["unfingered"]} / {r["sideways"]} / {r["short"]} / {r["shallow"]}',
                  f'{r["wrist_dev_median"]} / {r["wrist_dev_p95"]}', r["jitter_mm"], r["tremor_per_joint_s"]])
queued = [e for e in E if e not in {r["id"] for r in rows}]
m = json.load(open(os.path.join(W, "manifest.json")))
m["sections"] = [s for s in m["sections"] if s["id"] != "lab"]
ts = "?" if not rows else rows[-1]["started"]
m["sections"].insert(0, {"id": "lab", "title": "Piano-hand lab: every note, every key (target 100% key sync)",
  "intro": ("Each experiment solves the whole suite (Clair de Lune -> Rachmaninoff -> Chopin complete, ~1,820 notes) in 10 s windows "
            "on toymaker + sparky + turbo and scores every note: key at least half down within 0.25 s of its onset. Legs: H0 baseline, "
            "H1 ragdoll-first (loosen to 100%, then reverse into realism), H2 data-driven real-size hand (FürElise), H3 ablations. "
            f"Notebook: lab/NOTEBOOK.md in newjordan/h3_band_TFCV (ks-wrist). Last run started {ts}; queued: {', '.join(queued) or 'none'}."),
  "items": [{"kind": "table", "title": "Results (whole suite)", "rows": table}] +
           [{"kind": "image", "src": f"media/lab/{f}", "title": t} for f, t in (
               ("keysync_runs.png", "Key sync by experiment"), ("keysync_sections.png", "Key sync per section"),
               ("misses.png", "Missed notes by kind"), ("tradeoff_wrist.png", "Key sync vs wrist deviation"),
               ("tradeoff_jitter.png", "Key sync vs wrist shake"), ("tradeoff_tremor.png", "Key sync vs finger tremor"),
               ("k_handfidelity.png", "Leg K: H3 render vs Blender plate inside the hands, per run"),
               ("k_frame_vs_hand.png", "Leg K: whole-frame vs hand-region structure"))
            if os.path.exists(os.path.join(LAB, "plots", f))]})
json.dump(m, open(os.path.join(W, "manifest.json"), "w"), indent=1)
subprocess.run(["python3", os.path.expanduser("~/h3/mv_tools/band_site.py")], capture_output=True)
print("published", len(rows), "rows")
