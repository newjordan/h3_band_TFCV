#!/usr/bin/env python3
"""Drum-stem hit identifier: onsets classified into kit pieces, for blockout/drums.py.

Candidate onsets are the union of spectral-flux picks (same STFT as guitar_events.py) in the sub,
low, noise, 3-7 kHz and 8-16 kHz bands. At each one, band energy rises (dB over the 45 ms before)
and band levels (dB under the band's 99.5th-percentile level, so quiet bleed is ignored) decide:
  kick    35-90 Hz rise, sub louder than 110-300 Hz afterwards (a small rise is enough on a sub
          flux pick, for kicks over the previous kick's tail)
  snare   1.5-5 kHz rise with 160-240 Hz body, noise band still loud 25-60 ms later (wires)
  toms    110-300 Hz rise without a kick or snare; floor / tom2 / tom1 by the 60-400 Hz centroid
  metal   8-16 kHz rise, split by decay rate and brightness: closed / open hi-hat, ride, crash;
          plus ride hits that only show as 3-7 kHz flux while the cymbal is already ringing
Velocity is the deciding rise, scaled per piece to 40..127 against its 95th percentile.

It is a heuristic tuned on blockout.drumsynth audio, not a transcriber. Recall / precision within
30 ms on two synthesized grooves (examples/drums/rock_groove.mid at 112 bpm, and drum_groove
--seed 7 --bpm 128): kick 0.81-0.92 / 0.88, snare 0.68-0.76 / 0.89-1.0, hi-hat 0.85 / 0.94-1.0,
crash 0.75-0.88 / 1.0, ride 0.59-0.61 / 0.61-0.74, toms 0.33-0.83 / 0.27-1.0 (unreliable: the
piece is right more often than which tom). Check it against a known hit list with --truth before
trusting it on a real stem.

Usage: drum_events.py DRUMS.wav OUT.json [--truth HITS.json]
"""
import argparse, json

import numpy as np

from guitar_events import band_flux, pick, spectrum

TOL = 0.03


def _near(t, ts, tol=TOL):
    if not len(ts):
        return False
    i = np.searchsorted(ts, t)
    return any(0 <= j < len(ts) and abs(ts[j] - t) <= tol for j in (i - 1, i))


def _vel(x):
    x = np.asarray(x, float)
    if not len(x):
        return x
    p = max(np.percentile(x, 95), 1e-9)
    return np.round(40 + 87 * np.clip(x / p, 0, 1) ** 0.7).astype(int)


BANDS = {"sub": (35, 90), "low": (110, 300), "body": (160, 240), "noise": (1500, 5000), "mid": (3000, 7000),
         "hi": (8000, 16000), "all": (35, 16000)}
RISE_DB = {"sub": 8.0, "noise": 8.0, "body": 4.0, "low": 8.0, "hi": 8.0}
# a hit's band level must come within this many dB of the band's loud level (99.5th pct), so bleed
# from other pieces that merely rises from silence does not count
GATE_DB = {"sub": 18.0, "noise": 16.0, "low": 18.0, "hi": 20.0}
RIDE_PING_DB = 2.0
SNARE_SUSTAIN_DB = 12.0


def _db(mag, freqs, lo, hi):
    b = (freqs >= lo) & (freqs < hi)
    p = (mag[b] ** 2).sum(axis=0)
    return 10 * np.log10(p + 1e-7 * p.max() + 1e-20)


def detect(path):
    mag, freqs, fps, t0 = spectrum(path)
    E = {k: _db(mag, freqs, *v) for k, v in BANDS.items()}
    n = mag.shape[1]
    ms = lambda x: int(round(x * fps / 1000))

    def win(e, i, a, b):
        return e[min(max(i + ms(a), 0), n - 1):min(max(i + ms(b), 1), n)]

    peak = {k: float(np.percentile(E[k], 99.5)) for k in E}

    def rise(k, i, a=3, b=30):
        pre, post = win(E[k], i, -45, -5), win(E[k], i, a, b)
        return float(post.max() - pre.mean()) if len(pre) and len(post) else 0.0

    def loud(k, i):
        post = win(E[k], i, 0, 30)
        return len(post) and float(post.max()) >= peak[k] - GATE_DB[k]

    # candidate onsets: union of band-flux peaks, merged within 25 ms
    picks = {k: pick(band_flux(mag, freqs, *BANDS[k]), fps, 1.5) for k in ("sub", "low", "noise", "mid", "hi")}
    ping, sub_pk, low_pk = (np.array(sorted(picks[k])) for k in ("mid", "sub", "low"))
    on = []
    for i in sorted(np.concatenate(list(picks.values()))):
        if not on or i - on[-1] > ms(25):
            on.append(int(i))
    hits = []
    for c, i in enumerate(on):
        t = t0 + i / fps
        r = {k: rise(k, i) for k in RISE_DB}
        nxt = on[c + 1] if c + 1 < len(on) else n
        # a kick right after another rises only a few dB over the first one's tail, so a sub-band flux
        # pick is enough; a huge rise is a kick even when a tom keeps the low band up
        sub_over_low = win(E["sub"], i, 30, 60).mean() - win(E["low"], i, 30, 60).mean()
        kick = loud("sub", i) and (
            (sub_over_low > 0 and (r["sub"] >= RISE_DB["sub"] or (r["sub"] >= 3 and _near(i, sub_pk, ms(25)))))
            or (r["sub"] >= 20 and sub_over_low > -8))
        if kick:
            hits.append({"t": t, "piece": "kick", "_v": r["sub"]})
        # snare wires keep the noise band loud for tens of ms; a hat tick does not. A kick under a hat
        # lifts 110-300 Hz well above the snare body band.
        sustain = win(E["noise"], i, 25, 60)
        snare = (r["noise"] >= RISE_DB["noise"] and r["body"] >= RISE_DB["body"] and loud("noise", i)
                 and len(sustain) and sustain.max() >= peak["noise"] - SNARE_SUSTAIN_DB
                 and r["low"] - r["body"] < 3.3 and r["hi"] < r["noise"] + 6)
        if snare:
            hits.append({"t": t, "piece": "snare", "_v": r["noise"], "rise_db": {k: round(v, 1) for k, v in r.items()}})
        elif (not kick and loud("low", i)
              and (r["low"] >= RISE_DB["low"] or (r["low"] >= 3 and _near(i, low_pk, ms(25))))):
            b = (freqs >= 60) & (freqs < 400)
            seg = (mag[b, i:i + ms(80)] ** 2).sum(axis=1)
            cen = float((freqs[b] * seg).sum() / max(seg.sum(), 1e-20))
            piece = "floor" if cen < 150 else "tom2" if cen < 192 else "tom1"
            hits.append({"t": t, "piece": piece, "_v": r["low"], "centroid_hz": round(cen, 1)})
        if r["hi"] >= RISE_DB["hi"] and loud("hi", i):
            seg = win(E["hi"], i, 0, min(150, 1000 * (nxt - i) / fps - 5))
            pk = int(np.argmax(seg[:ms(30) + 1]))
            tail = seg[pk:]
            slope = float((tail[-1] - tail[0]) / max(len(tail) / fps * 10, 1e-3)) if len(tail) > ms(20) else -99.0
            bright = float(win(E["hi"], i, 0, 20).max() - win(E["mid"], i, 0, 20).max())
            band = (freqs >= 2500) & (freqs < 7000)
            spec = (mag[band, i + ms(10):i + ms(40)] ** 2).mean(axis=1)
            peaky = float(10 * np.log10(spec.max() / max(spec.mean(), 1e-20))) if spec.size else 0.0
            h = {"t": t, "_v": r["hi"], "decay_db_per_100ms": round(slope, 1), "bright_db": round(bright, 1),
                 "peaky_db": round(peaky, 1)}
            # decay (dB / 100 ms): closed hat ~-24, hat under a snare tail ~-10, ride ~-6, crash > -5
            if slope < -12:
                h["piece"] = "hihat"
            elif bright > 17:
                h["piece"], h["open"] = "hihat", True
            elif slope < -8:
                h["piece"] = "hihat"
            elif slope > -4.5 or bright > 14:
                h["piece"] = "crash"
            else:
                h["piece"] = "ride"
            hits.append(h)
        elif (not snare and _near(i, ping, ms(25)) and rise("mid", i) >= RIDE_PING_DB
              and win(E["hi"], i, -45, -5).mean() >= peak["hi"] - GATE_DB["hi"]):
            # a ride in a groove rings over itself, so its hi band barely rises; the stick ping still
            # shows as 3-7 kHz flux while the hi band is already up
            hits.append({"t": t, "piece": "ride", "_v": rise("mid", i), "ping_only": True})
    for p in {h["piece"] for h in hits}:
        hs = [h for h in hits if h["piece"] == p]
        for h, v in zip(hs, _vel([h["_v"] for h in hs])):
            h["vel"] = int(v)
            del h["_v"]
            h["t"] = round(h["t"], 4)
    return sorted(hits, key=lambda h: h["t"])


def _alias(p):
    return "crash" if p == "crash2" else p


def compare(hits, truth):
    """Per piece precision / recall within TOL. Truth 'hihat_pedal' is skipped (feet chicks are quiet);
    'crash2' scores as 'crash' since the detector cannot tell two crashes apart."""
    out = {}
    pieces = sorted({_alias(h["piece"]) for h in truth if h["piece"] != "hihat_pedal"} | {h["piece"] for h in hits})
    for p in pieces:
        tr = np.array(sorted(h["t"] for h in truth if _alias(h["piece"]) == p))
        de = np.array(sorted(h["t"] for h in hits if h["piece"] == p))
        tp_r = sum(_near(t, de) for t in tr)
        tp_p = sum(_near(t, tr) for t in de)
        out[p] = {"truth": len(tr), "found": len(de),
                  "recall": round(tp_r / len(tr), 3) if len(tr) else None,
                  "precision": round(tp_p / len(de), 3) if len(de) else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drums"); ap.add_argument("out")
    ap.add_argument("--truth", help="known hit list JSON to score against")
    a = ap.parse_args()
    hits = detect(a.drums)
    summary = {p: sum(h["piece"] == p for h in hits) for p in sorted({h["piece"] for h in hits})}
    res = {"source": f"band-flux onsets on {a.drums}", "method": __doc__.strip(), "summary": summary, "hits": hits}
    if a.truth:
        res["vs_truth"] = compare(hits, json.load(open(a.truth))["hits"])
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("summary", "vs_truth") if k in res}, indent=1))


if __name__ == "__main__":
    main()
