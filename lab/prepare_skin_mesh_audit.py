"""Prepare actual Blender triangles with independently verified side labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from blockout.skin import Skin
from blockout.subdivision import catmull_clark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parity_report', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    parity = json.loads(args.parity_report.read_text())
    if not parity['passed'] or parity['validation'] != 'complete' or parity['subdivision_levels'] != 2:
        raise ValueError('Completed level-2 full topology/vertex parity is required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for record in parity['records']:
        actual_file = Path(record['data_file'])
        with np.load(actual_file, allow_pickle=False) as data:
            vertices, triangles = data['blender'], data['blender_triangles']
            expected, sides, regions = [], [], []
            for side in ('L', 'R'):
                skin = Skin(side, scale=parity['hand_scale'])
                operator, _ = catmull_clark(len(skin.vertices), skin.faces, 2, limit=True)
                expected.append(operator @ data[side])
                sides.extend([side] * operator.shape[0])
                names = [skin.names[i] for i in np.asarray(operator @ skin.weights).argmax(1)]
                regions.extend([side + ':' + (n.split('-')[0] if n.startswith('finger') else n) for n in names])
            expected = np.concatenate(expected)
        error, mapping = cKDTree(expected).query(vertices)
        if (len(vertices) != len(expected) or not np.array_equal(np.sort(mapping), np.arange(len(expected)))
                or error.max() * 1000. > parity['tolerance_mm'] or not record['surface_faces_match']):
            raise ValueError('Actual rendered vertices do not map bijectively to the validated hand surfaces')
        out = args.output.with_name(f'{args.output.stem}_frame{record["frame"]}.npz')
        np.savez_compressed(out, vertices=vertices, triangles=triangles,
            vertex_side=np.asarray(sides)[mapping], vertex_region=np.asarray(regions)[mapping])
        records.append(dict(frame=record['frame'], mesh=str(out.resolve()),
            mesh_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
            actual_blender_mesh_sha256=hashlib.sha256(actual_file.read_bytes()).hexdigest(),
            mapping_maximum_error_mm=float(error.max() * 1000.), vertices=len(vertices), triangles=len(triangles)))
    manifest = dict(schema='actual-rendered-skin-mesh-audit-input/v1',
        source=parity['source'], hand_scale=parity['hand_scale'], records=records,
        parity_report=str(args.parity_report.resolve()),
        parity_report_sha256=hashlib.sha256(args.parity_report.read_bytes()).hexdigest(),
        code_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
            ('blockout/skin.py', 'blockout/subdivision.py', 'blockout/hand_model/skin_bind.npz', 'lab/prepare_skin_mesh_audit.py')},
        label_scope='dominant interpolated bone region is for locating witnesses, not a collision exemption')
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
