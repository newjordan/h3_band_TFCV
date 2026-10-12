import mido,collections
m=mido.MidiFile('raw.mid');print(m.type,m.ticks_per_beat,len(m.tracks),m.length)
notes=[];on={};t=0;ped=[]
for msg in mido.merge_tracks(m.tracks):
    t+=msg.time
    if msg.type=='set_tempo':print('tempo',msg.tempo)
    s=mido.tick2second(t,m.ticks_per_beat,500000)
    if msg.type=='note_on' and msg.velocity>0:on[msg.note]=(s,msg.velocity)
    elif msg.type in('note_off','note_on') and msg.note in on:
        a,v=on.pop(msg.note);notes.append((a,s,msg.note,v))
    if msg.type=='control_change' and msg.control==64:ped.append((s,msg.value))
print(len(notes),len(ped),notes[0],max(n[1] for n in notes))
pc=collections.Counter(n[2]%12 for n in notes);print(sorted(pc.items(),key=lambda x:-x[1]))
print(sorted(collections.Counter(n[3]//10*10 for n in notes).items()))
print('short',sum(1 for n in notes if n[1]-n[0]<0.06))
import pickle;pickle.dump((notes,ped),open('notes.pkl','wb'))
for n in sorted(notes)[:70]:print(f"{n[0]:7.2f} {n[1]-n[0]:5.2f} {n[2]} {mido.second2tick and ''}{n[3]}")
