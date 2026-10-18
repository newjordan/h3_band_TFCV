import json, sys, numpy as np
from blockout.piano import key_x, is_black
S="/tmp/claude-1000/-home-frosty40-h3/b1370b84-c277-4ad5-9e80-bcb80177e714/scratchpad"
for tag in sys.argv[1:]:
    a=json.load(open(f"{S}/{tag}.json")); fr=a["frames"]; fps=a["fps"]; st=a["start"]
    notes=json.load(open(f"{S}/{tag}_notes.json")); fing=a["fingering"]
    fmap={}
    for h,sl in fing.items():
        for s in sl:
            for t,p,f in s: fmap[(round(t,3),p)]=(h,f)
    print("##",tag)
    for s,e,p,v,tr in notes:
        t=s-st
        if t<0 or t>len(fr)/fps-0.3: continue
        j0,j1=int(t*fps),min(len(fr)-1,int((min(e,s+0.25)-st)*fps)+1)
        dd=[fr[j]['keys'].get(str(p),0.0) for j in range(j0,j1+1)]; d=max(dd)
        if d<0.5:
            h,f=fmap.get((round(s,3),p),("?",0))
            jj=j0+int(np.argmin([abs(fr[j]["hands"][h]["pads"][f][2]) for j in range(j0,j1+1)]))
            pad=np.array(fr[jj]["hands"][h]["pads"][f])
            print(f" miss t={t:.2f} p={p} {'blk' if is_black(p) else 'wht'} {h}{f+1} depth {d:.2f} pad-key dx {1000*(pad[0]-key_x(p)):.0f} mm y {1000*pad[1]:.0f} z {1000*pad[2]:.0f}  wrist {np.round(np.array(fr[jj]['hands'][h]['wrist'])*1000).astype(int)}")
