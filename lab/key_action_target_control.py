"""Compare all prepared control inputs with the preceding committed controller."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
ART=ROOT/'artifacts/realism'
import numpy as np
from blockout import feeling,midi,piano


def canonical(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,midi.Note):
        return dict(note=[value.start,value.end,value.pitch,value.velocity,value.track],
            touch=getattr(value,'touch',None),lift=getattr(value,'lift',None))
    if isinstance(value,(tuple,list)):return [canonical(v) for v in value]
    if isinstance(value,dict):return {str(k):canonical(v) for k,v in value.items() if k!='action_trace'}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if hasattr(value,'__dict__'):return dict(type=type(value).__name__,state=canonical(vars(value)))
    raise TypeError(type(value))


def main():
    source=subprocess.check_output(['git','show','e260fc4dc4cd692820a2dad90baa0088818476b7:blockout/piano.py'])
    path=ART/'key_action_legacy_piano.py';path.write_bytes(source)
    spec=importlib.util.spec_from_file_location('blockout.key_action_legacy_piano',path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old
    old.__file__=str(ROOT/'blockout/piano.py') # restore the committed module's relative asset context
    spec.loader.exec_module(old)
    for module in (old,piano):module.REACH_SPLIT=True
    source_midi=ROOT/'examples/piano/knights_suite.mid'
    notes=midi.read(source_midi);beats,db=midi.beats(source_midi)
    chart=feeling.Chart(str(source_midi.with_suffix('.feeling.json')),db)
    notes,timing=feeling.perform(notes,chart,0)
    scores=[m.compile_score(notes) for m in (old,piano)]
    records=[]
    for start,duration in ((282.9,4.),(118.9,4.),(127.9,2.)):
        plans=[m.prepare_plan(compiled_score=s,start=start,dur=duration,fps=24,physical_warmup=True,
            beats=[float(timing(t)) for t in beats],downbeats=[float(timing(t)) for t in db],
            schedule=[(float(timing(t)),style) for t,style in chart.heads],
            energy_fn=lambda t:float(np.clip(1.4*chart.at('strength',np.interp(t,timing.tp,timing.ts)),0,1)),seed=0)
            for m,s in zip((old,piano),scores)]
        fields=[field.name for field in dataclasses.fields(plans[0])]
        a={field:canonical(getattr(plans[0],field)) for field in fields}
        b={field:canonical(getattr(plans[1],field)) for field in fields}
        assert a==b,(start,[field for field in fields if a[field]!=b[field]])
        payload=json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        records.append(dict(start=start,duration=duration,internal_frames=len(plans[0].times),
            all_preexisting_fields_equal=True,prepared_input_sha256=hashlib.sha256(payload).hexdigest()))
    report=dict(status='complete',legacy_commit='e260fc4dc4cd692820a2dad90baa0088818476b7',
        legacy_module_sha256=hashlib.sha256(source).hexdigest(),records=records,
        asset_context=str(ROOT/'blockout/piano.py'),
        scope='All preexisting prepared inputs, including body spring state; only new lineage metadata omitted. '
            'This does not reverify solved skin trajectories or accept performance.')
    (ART/'key_action_target_control.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
