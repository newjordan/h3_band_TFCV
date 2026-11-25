"""Tiny Standard MIDI writer shared by the reader tests."""
import struct
from pathlib import Path


def _vlq(v):
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append(0x80 | (v & 0x7F))
        v >>= 7
    return bytes(reversed(out))


def write_midi(path, events, division=480):
    """events are (absolute_tick, raw bytes). A running-status event omits its status byte."""
    track, last = b"", 0
    for tick, data in events:
        track += _vlq(tick - last) + data
        last = tick
    track += _vlq(0) + b"\xff\x2f\x00"
    blob = b"MThd" + struct.pack(">IHHH", 6, 0, 1, division) + b"MTrk" + struct.pack(">I", len(track)) + track
    Path(path).write_bytes(blob)
