"""Second-order dynamics: the difference between procedural motion that looks robotic and motion that looks alive.

A target trajectory (where a hand "wants" to be) is fed through a damped spring:
    y + k1 y' + k2 y'' = x + k3 x'
with k1 = z / (pi f), k2 = 1 / (2 pi f)^2, k3 = r z / (2 pi f).
  f  natural frequency (Hz): how fast it responds
  z  damping: < 1 overshoots and settles (follow-through), 1 is critical, > 1 is sluggish
  r  initial response: 0 eases in, 1 is immediate, > 1 overshoots at the start, < 0 anticipates (winds up)
Integrated with semi-implicit Euler and the k2 stability clamp, several substeps per frame.
"""
import math

import numpy as np


class SecondOrder:
    def __init__(self, f, z, r, x0):
        self.k1 = z / (math.pi * f)
        self.k2 = 1 / (2 * math.pi * f) ** 2
        self.k3 = r * z / (2 * math.pi * f)
        self.xp = np.array(x0, float)
        self.y = np.array(x0, float)
        self.yd = np.zeros_like(self.y)

    def step(self, T, x):
        x = np.asarray(x, float)
        xd = (x - self.xp) / T
        self.xp = x
        k2 = max(self.k2, T * T / 2 + T * self.k1 / 2, T * self.k1)
        self.y = self.y + T * self.yd
        self.yd = self.yd + T * (x + self.k3 * xd - self.y - self.k1 * self.yd) / k2
        return self.y


def filter_track(x, fps, f, z, r, substeps=4):
    """Run a whole (N, ...) target track through SecondOrder; targets are linearly interpolated between frames."""
    x = np.asarray(x, float)
    out = np.empty_like(x)
    so = SecondOrder(f, z, r, x[0])
    T = 1.0 / (fps * substeps)
    out[0] = x[0]
    for i in range(1, len(x)):
        for s in range(1, substeps + 1):
            y = so.step(T, x[i - 1] + (x[i] - x[i - 1]) * s / substeps)
        out[i] = y
    return out


def drift(times, amp, seed, rates=(0.13, 0.31, 0.71)):
    """Slow organic wander (sum of incommensurate sines with seeded phases), amplitude ~amp."""
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 2 * np.pi, len(rates))
    w = np.array([1.0, 0.5, 0.25])[: len(rates)]
    return amp * sum(wi * np.sin(2 * np.pi * ri * times + p) for wi, ri, p in zip(w, rates, ph)) / w.sum()
