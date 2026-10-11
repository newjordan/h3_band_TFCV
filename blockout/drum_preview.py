"""Fixed-camera CPU preview of drummer blockout JSON; PNG contact sheet or MP4.

Example: python -m blockout.drum_preview /tmp/drums.json --out /tmp/drums.mp4
The colored finger chains are contact proxies, not the production hand mesh.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

from .drums import unit


COLORS = {"L": "#25b6e0", "R": "#f08b51"}


def draw(ax, data, frame):
    ax.clear()
    ax.set_facecolor("#111b29")
    for drum in data["kit"]:
        normal = np.array(drum["normal"])
        tangent = unit(np.cross(normal, np.array([0., 1., 0.])))
        across = np.cross(normal, tangent)
        theta = np.linspace(0, 2*np.pi, 70)
        points = np.asarray(drum["center"])+drum["radius"]*(np.cos(theta)[:, None]*tangent+np.sin(theta)[:, None]*across)
        ax.plot(*points.T, color="#71839b", lw=1.2)
        ax.text(*drum["center"], drum["name"], color="#c0cbda", fontsize=7)
    shoulders = [frame["hands"][h]["shoulder"] for h in ("L", "R")]
    ax.plot(*np.asarray(shoulders).T, color="#b1bed0", lw=6)
    for hand, pose in frame["hands"].items():
        color = COLORS[hand]
        arm = np.array([pose["shoulder"], pose["elbow"], pose["wrist"], pose["grip"]])
        ax.plot(*arm.T, color=color, linewidth=4, marker="o", markersize=4)
        stick = np.array([pose["butt"], pose["tip"]])
        ax.plot(*stick.T, color="#e4d6b6", linewidth=2)
        for finger in pose["fingers"]:
            ax.plot(*np.asarray(finger).T, color=color, lw=2)
        ax.scatter(*pose["tip"], s=22, color="#ffff99" if pose["phase"] == "contact" else color)
    ax.set(xlim=(-.58, .65), ylim=(-.40, .70), zlim=(0., .67))
    ax.set_box_aspect((1.23, 1.10, .67), zoom=1.2)
    ax.set_proj_type("ortho")
    ax.view_init(elev=27, azim=-62)
    ax.set_axis_off()
    pressure = "  ".join(f"{h} grip {p['pressure']:.2f}" for h, p in frame["hands"].items())
    ax.text2D(.04, .94, f"{frame['time']:.3f}s   {pressure}", transform=ax.transAxes,
              color="#e2e8f0", fontsize=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("anim")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data = json.load(open(args.anim))
    out = Path(args.out)
    if out.suffix.lower() == ".mp4":
        fig = plt.figure(figsize=(9.6, 7.2), facecolor="#111b29")
        ax = fig.add_subplot(111, projection="3d")
        writer = FFMpegWriter(fps=data["fps"], bitrate=1800)
        with writer.saving(fig, str(out), dpi=100):
            for frame in data["frames"]:
                draw(ax, data, frame)
                writer.grab_frame()
    elif out.suffix.lower() == ".png":
        fig = plt.figure(figsize=(15, 12), facecolor="#111b29")
        times = [e["time"] for e in data["events"]]
        selected = (times[:9] if times else np.linspace(0, data["duration"], 9))
        for i, t in enumerate(selected):
            frame = min(data["frames"], key=lambda f: abs(f["time"]-t))
            draw(fig.add_subplot(3, 3, i+1, projection="3d"), data, frame)
        fig.suptitle("Drummer contact controller · fixed camera · grip proxies", color="#e2e8f0",
                     fontsize=17, y=.985, zorder=100)
        fig.subplots_adjust(left=.01, right=.99, bottom=0, top=.91, wspace=.02, hspace=0)
        fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    else:
        parser.error("--out must end in .png or .mp4")
    plt.close("all")
    print(out)


if __name__ == "__main__":
    main()
