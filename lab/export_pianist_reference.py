"""Export compact soft priors from the local FürElise statistics file."""
import argparse
import hashlib
import json
from pathlib import Path


def export(source):
    data = json.loads(source.read_text())
    return dict(schema='pianist-geometric-priors/v1', source=data['source'],
        source_url='https://arxiv.org/html/2410.05791v1',
        source_statistics_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        held_out=data['held_out'], pieces=len(data['pieces']), n_hand_frames=data['n_hand_frames'],
        phase_bands=data['press_split'], per_finger=data['per_finger'],
        strike=data['strike'], placement=data['placement'], limitations=[
            'markerless MANO reconstruction and MIDI-based refinement, not force ground truth',
            'phase bands pool fingers and subjects; no per-finger phase conditioning',
            'pooled depth priors are not individual-pianist or per-key distributions',
            'bone landmarks and actual skin contact points differ'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(export(args.source), indent=2) + '\n')
