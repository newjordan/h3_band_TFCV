"""Check Blender's exported subdivided skin against NumPy surface stencils.

python lab/skin_parity_check.py artifacts/realism/skin_parity_overlap_subdiv2.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blockout.skin import Skin
from blockout.subdivision import catmull_clark


def check(path):
    report = json.loads(path.read_text())
    level = report['subdivision_levels']
    skins = {side: Skin(side, scale=report['hand_scale']) for side in ('L', 'R')}
    surfaces = {side: catmull_clark(len(skin.vertices), skin.faces, level,
                                  limit=report.get('limit_surface', True))
                for side, skin in skins.items()}
    operators = {side: surface[0] for side, surface in surfaces.items()}
    offset, all_faces = 0, []
    for side in skins:
        all_faces.append(surfaces[side][1] + offset)
        offset += operators[side].shape[0]
    faces = np.concatenate(all_faces)
    for record in report['records']:
        with np.load(record['data_file'], allow_pickle=False) as data:
            actual = np.concatenate([operators[side] @ data[side] for side in skins])
            expected = data['blender']
            blender_faces = data['blender_faces'] if 'blender_faces' in data else None
        record['topology_count_matches'] = len(actual) == len(expected)
        # Blender's OpenSubdiv orders vertices differently. Test both complete
        # point sets, with an additional exact vertex-count requirement.
        forward, mapping = cKDTree(actual).query(expected)
        error = np.r_[forward, cKDTree(expected).query(actual)[0]] * 1000.
        if blender_faces is not None:
            def ordered(polygons):
                polygons = np.sort(polygons, axis=1)
                return polygons[np.lexsort(polygons.T[::-1])]
            record['vertex_mapping_bijective'] = np.array_equal(np.sort(mapping), np.arange(len(actual)))
            record['surface_faces_match'] = np.array_equal(ordered(faces), ordered(mapping[blender_faces]))
        else:
            record['surface_faces_match'] = None
        record.update(maximum_error_mm=float(error.max()),
                      rms_error_mm=float(np.sqrt(np.mean(error ** 2))))
    report['passed'] = all(r['topology_count_matches'] and
                           r.get('surface_faces_match') is True and r.get('vertex_mapping_bijective') is True and
                           r['maximum_error_mm'] < report['tolerance_mm'] and
                           r.get('scene_box_maximum_error_mm', float('inf')) < report['tolerance_mm']
                           for r in report['records'])
    report['validation'] = 'complete'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report['passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if check(args.report) else 2)
