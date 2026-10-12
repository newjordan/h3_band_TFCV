import sys,librosa
from piano_transcription_inference import PianoTranscription, sample_rate
a,_=librosa.load('/tmp/claude-1000/suite/work/levy16k.wav',sr=sample_rate)
t=PianoTranscription(device='cuda',checkpoint_path='/tmp/claude-1000/suite/work/model.pth')
t.transcribe(a,'/tmp/claude-1000/suite/work/raw.mid')
