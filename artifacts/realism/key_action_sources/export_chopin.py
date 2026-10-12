import json
from blockout import midi, suite
d=json.load(open('/tmp/claude-1000/suite/work/fit.json'));b=d['bars'];t0=b[0]
notes=[midi.Note(s-t0,e-t0,p,v,1) for s,e,p,v in d['notes'] if s>=t0-0.05]
notes=[midi.Note(max(n.start,0),n.end,n.pitch,n.velocity,1) for n in notes]
bars=[(b[i]-t0,b[i+1]-b[i],4,4) for i in range(64)]
suite.write('examples/piano/chopin_nocturne_cs_minor_posth.mid',suite.resplit(notes),bars)
