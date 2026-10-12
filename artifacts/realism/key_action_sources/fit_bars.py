import pickle,numpy as np,json
notes,ped=pickle.load(open('notes.pkl','rb'))
# ghost filter: very short AND quiet
keep=[n for n in notes if not ((n[1]-n[0])<0.08 and n[3]<40) and (n[1]-n[0])>0.03]
print(len(notes),'->',len(keep))
keep.sort()
end=max(n[1] for n in keep)
on=np.array(sorted({round(n[0],2) for n in keep}))
bass=np.array(sorted(n[0] for n in keep if n[2]<=52))
# candidate cluster: onsets (merge within 60 ms)
cand=[on[0]]
for t in on[1:]:
    if t-cand[-1]>0.06:cand.append(t)
cand=np.array(cand)
def cost_c(t):
    d=np.min(np.abs(bass-t)); return 0.0 if d<0.08 else 1.0
cc=np.array([cost_c(t) for t in cand])
# last sounding chord: final onset; bar 64 ends at 'end'
N=64
first=0
lastc=len(cand)-1
INF=1e18
# best[k][j]: min cost with boundary k (0..N) at cand j; boundary 0 = cand[0]; boundary N = virtual end
best=np.full((N+1,len(cand)),INF);arg=np.zeros((N+1,len(cand)),int)
best[0][0]=0
# need prev length for smoothness: approximate by using only pairwise log ratio with predecessor's own best predecessor (greedy)
plen=np.full((N+1,len(cand)),np.nan)
for k in range(1,N+1):
    for j in range(1,len(cand)):
        bs=INF
        for i in range(0,j):
            if best[k-1][i]>=INF:continue
            L=cand[j]-cand[i]
            if L<1.4 or L>7.5:continue
            pl=plen[k-1][i]
            sm=0 if np.isnan(pl) else 6*np.log(L/pl)**2
            c=best[k-1][i]+cc[j]*1.0+sm
            if c<bs:bs=c;arg[k][j]=i;plen[k][j]=L
        best[k][j]=bs
# final: boundary N is the end of bar N = 'end' (last note end); choose last bar start j = boundary N-1; add length cost
bestj=None;bc=INF
for j in range(1,len(cand)):
    if best[N-1][j]>=INF:continue
    L=end-cand[j]
    if L<1.4 or L>7.5:continue
    c=best[N-1][j]
    if c<bc:bc=c;bestj=j
idx=[bestj]
for k in range(N-1,0,-1):idx.append(arg[k][idx[-1]])
b=[cand[i] for i in reversed(idx)]+[end]
b=[float(x) for x in b]
print(len(b)-1,'bars cost',bc,'bass-misses',sum(cc[i] for i in idx))
L=np.diff(b);print(np.round(L,2))
json.dump({'bars':b,'notes':keep},open('fit.json','w'))
