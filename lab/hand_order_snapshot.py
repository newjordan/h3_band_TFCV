"""Extract unchanged production skin for independent crossing diagnostics."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from blockout.skin import Skin


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    arguments=sys.argv[sys.argv.index('--')+1:]
    source,directory=map(Path,arguments[:2])
    indices=list(map(int,arguments[2].split(',')))
    animation=json.loads(source.read_text())
    directory.mkdir(parents=True,exist_ok=True)
    fingerprints={str(p):sha(p) for p in [source,ROOT/'blockout/blender_piano.py',ROOT/'blockout/skin.py',
                                         ROOT/'lab/hand_order_snapshot.py']}
    sys.argv=['blender','--',str(source),str(directory),'--frames','0:0',
              '--res','768x432','--quality','draft','--hide','body']
    renderer=runpy.run_path(str(ROOT/'blockout/blender_piano.py'),run_name='hand_order_renderer')
    arm,rest=renderer['MPFB_ARM'],renderer['MPFB_REST']
    mesh=next(o for o in bpy.context.scene.objects if o.type=='MESH' and any(m.type=='ARMATURE' for m in o.modifiers))
    assert not next(m for m in mesh.modifiers if m.type=='ARMATURE').use_deform_preserve_volume
    smooth=next(m for m in mesh.modifiers if m.type=='SUBSURF')
    smooth.levels=smooth.render_levels
    assert smooth.levels==2 and smooth.use_limit_surface
    report=dict(status='running',hand_scale=animation['hand_scale'],candidate_exported=False,records=[],
        input_sha256=fingerprints,scope='Matched actual skin snapshots only; no full trajectory acceptance')
    for index in indices:
        frame=animation['frames'][index]
        for side in ('L','R'):
            renderer['pose_rig'](arm,rest,side,frame['hands'][side])
        for pitch,key in renderer['keys'].items():
            key.rotation_euler=(float(frame['keys'].get(pitch,0.))*.010/key['length'],0.,0.)
        bpy.context.view_layer.update()
        evaluated=mesh.evaluated_get(bpy.context.evaluated_depsgraph_get())
        value=evaluated.to_mesh(); value.calc_loop_triangles()
        vertices=np.array([evaluated.matrix_world @ v.co for v in value.vertices])
        faces=np.array([p.vertices[:] for p in value.polygons],np.int32)
        triangles=np.array([t.vertices[:] for t in value.loop_triangles],np.int32)
        evaluated.to_mesh_clear()
        target=directory/f'frame{index}.npz'
        np.savez_compressed(target,vertices=vertices,faces=faces,triangles=triangles,
            **{s:Skin(s,scale=animation['hand_scale']).from_frame(frame['hands'][s]) for s in ('L','R')})
        report['records'].append(dict(context=f'{source.stem}_frame{index}',frame=index,
            source=str(source.resolve()),source_sha256=sha(source),mode='linear',subdivision_levels=2,
            mesh=str(target.resolve()),mesh_sha256=sha(target),vertices=len(vertices),quads=len(faces),triangles=len(triangles)))
    report['source_unchanged_during_run']=all(sha(Path(p))==h for p,h in fingerprints.items())
    assert report['source_unchanged_during_run']
    report['status']='complete'
    (directory/'snapshots.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
