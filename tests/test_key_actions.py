import copy
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from blockout import feeling, midi, piano, sampler
from blockout.key_actions import BinaryPedalTimeline, legacy_sampler_damping


class KeyActionTests(unittest.TestCase):
    def test_timing_receipt_links_each_duplicate_pitch_without_changing_rng_or_source(self):
        class Chart:
            def at(self,key,time):
                return np.zeros_like(time,dtype=float)+{'tempo':1.,'strength':.5,'touch':.4,'freedom':.8}[key]
            def breath(self,time):return 0.
        notes=[midi.Note(0.,.01,60,70,1),midi.Note(.5,1.,60,70,1),midi.Note(.5,1.,60,70,1)]
        original=copy.deepcopy([vars(n) for n in notes])
        baseline,_=feeling.perform(notes,Chart(),seed=9)
        receipt={};output,_=feeling.perform(notes,Chart(),seed=9,timing_receipt=receipt)
        self.assertEqual([vars(n) for n in output],[vars(n) for n in baseline])
        self.assertEqual([vars(n) for n in notes],original)
        self.assertEqual(sorted(r['input_id'] for r in receipt['records']),[0,1,2])
        for record,note in zip(receipt['records'],output):
            self.assertEqual(record['output_note'],[note.start,note.end,note.pitch,note.velocity,note.track])
            self.assertAlmostEqual(record['shared_end']+record['actual_end_variation_seconds'],note.end)
        with self.assertRaises(ValueError):feeling.perform(notes,Chart(),timing_receipt=receipt)

    def test_control_changes_follow_tempo_and_running_status_without_extending_notes(self):
        def chunk(events):
            from blockout.suite import _vlq
            data=b''.join(_vlq(delta)+message for delta,message in events)+b'\0\xff\x2f\0'
            return b'MTrk'+struct.pack('>I',len(data))+data
        tempo=chunk([(0,b'\xff\x51\3\x07\xa1\x20'),(480,b'\xff\x51\3\x0f\x42\x40')])
        music=chunk([(0,b'\x90\x3c\x50'),(240,b'\xb1\x40\x3f'),(240,b'\x40\x40'),
                     (480,b'\x80\x3c\0')])
        other=chunk([(480,b'\xb2\x40\x7f'),(0,b'\x01\x2a')])
        data=b'MThd'+struct.pack('>IHHH',6,1,3,480)+tempo+music+other
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pedal.mid';path.write_bytes(data)
            before=midi.read(path)
            events=midi.control_changes(path,64)
            self.assertEqual([(e.time,e.value,e.channel,e.track) for e in events],
                             [(.25,63,1,1),(.5,64,1,1),(.5,127,2,2)])
            self.assertEqual(events[1].tick,480)
            self.assertEqual(len(midi.control_changes(path)),4)
            self.assertEqual(midi.read(path),before)
            self.assertEqual(before[0].end,1.5)
            with self.assertRaises(ValueError): midi.control_changes(path,128)

    def test_crop_inherits_pedal_and_threshold_63_64_without_fabricating_transitions(self):
        events=[(.2,64),(1.5,63),(2.,127)]
        timeline=BinaryPedalTimeline(events,start=1.,end=3.)
        self.assertTrue(timeline.state(1.))
        self.assertFalse(timeline.state(1.5))
        self.assertTrue(timeline.state(3.))
        self.assertEqual(timeline.events,tuple(events))
        with self.assertRaises(ValueError):timeline.state(.99)

    def test_early_release_requires_uninterrupted_sustain_and_honors_end_boundaries(self):
        timeline=BinaryPedalTimeline([(0.,127),(.8,0),(.9,127),(2.,0)],0.,3.)
        self.assertFalse(timeline.release_coverage(.7,1.2)['covered'])
        self.assertEqual(timeline.release_coverage(.7,1.2)['intervening_up_times'],[.8])
        self.assertTrue(timeline.release_coverage(.9,2.)['covered'])
        self.assertFalse(timeline.release_coverage(.8,.9)['covered'])
        self.assertTrue(timeline.release_coverage(2.,2.)['covered'])
        with self.assertRaises(ValueError):timeline.release_coverage(2.,1.)

    def test_pedal_extended_offset_yields_a_range_not_a_unique_key_release(self):
        timeline=BinaryPedalTimeline([(.2,127),(2.,0)],0.,3.)
        bounds=timeline.release_hypotheses(.5,2.)
        self.assertEqual(len(bounds),1)
        self.assertEqual((bounds[0]['earliest'],bounds[0]['latest']),(.5,2.))
        self.assertFalse(bounds[0]['unique'])
        self.assertEqual(timeline.release_hypotheses(.5,1.9),[])
        self.assertEqual(len(timeline.release_hypotheses(.5,1.99,.02)),1)

    def test_ambiguous_or_invalid_pedal_inputs_cannot_become_coverage(self):
        for events in ([(1.,127),(1.,0)],[(1.,127),(.5,0)],[(np.nan,127)],[(1.,.5)],[(1.,128)]):
            with self.subTest(events=events),self.assertRaises(ValueError):
                BinaryPedalTimeline(iter(events),0.,2.)
        with self.assertRaises(ValueError):BinaryPedalTimeline([],0.,np.inf)

    def test_legacy_sampler_damping_matches_actual_audio_calls_including_final_tail(self):
        notes=[[0.,.5,60,80,1],[0.,1.,40,70,2],[1.2,1.5,43,70,2],[2.,2.5,45,70,2]]
        called=[]
        class SpyPiano:
            def __init__(self,*args): pass
            def note(self,pitch,velocity,length):
                called.append((pitch,length))
                return np.ones((10,2),np.float32)
        with patch.object(sampler,'Piano',SpyPiano),patch.object(sampler,'SR',100):
            sampler.render(notes,'unused',dur=4.,wet=0.)
        resets=sampler.auto_pedal(notes)
        self.assertEqual(called,[(p,legacy_sampler_damping(e,resets)-s) for s,e,p,_,_ in notes])
        self.assertEqual(legacy_sampler_damping(2.5,resets),5.5)

    def test_every_identical_source_attack_keeps_an_id_and_dropped_attack_is_explicit(self):
        notes=[midi.Note(0.,1.,60,80,1),midi.Note(.02,1.,60,80,1)]
        original=copy.deepcopy([vars(n) for n in notes])
        score=piano.compile_score(notes)
        ledger=score.action_ledger
        self.assertEqual([r['source']['source_id'] for r in ledger['records']],[0,1])
        self.assertEqual(ledger['summary']['missing_source_ids'],[1])
        self.assertFalse(ledger['summary']['scheduling_bijection'])
        self.assertEqual(ledger['records'][1]['changes'][0]['stage'],'slicing')
        self.assertEqual([vars(n) for n in notes],original)

    def test_release_changes_have_stage_provenance_and_cannot_rewrite_source_duration(self):
        notes=[midi.Note(0.,2.,60,70,1),midi.Note(.4,.8,84,80,1)]
        with patch.object(piano,'NO_HAND_CROSS',False),patch.object(piano,'REACH_SPLIT',False):
            ledger=piano.compile_score(notes).action_ledger
        first=ledger['records'][0]
        self.assertEqual(first['source']['end'],2.)
        self.assertLess(first['scheduled_actions'][0]['end'],2.)
        stages={change['stage'] for change in first['changes']}
        self.assertTrue(stages&{'fingering_repair','finger_reuse','reach_release'})
        self.assertFalse(ledger['pedal_coverage_validated'])
        self.assertEqual(notes[0].end,2.)

    def test_rolled_copies_retain_source_ids_touch_and_lift(self):
        notes=[midi.Note(0.,1.,p,80,1) for p in (21,60,108)]
        for index,note in enumerate(notes):
            note._score_event_id=index;note.touch=.7;note.lift=.6
        with patch.object(piano,'ROLL_STAGGER',True): rows=piano.slices(notes,hand='R')
        copies=[n for row in rows for n in row]
        self.assertEqual(sorted(n._score_event_id for n in copies),[0,1,2])
        self.assertTrue(all(n.touch==.7 and n.lift==.6 for n in copies))
        self.assertEqual([n.start for n in notes],[0.,0.,0.])

    def test_compiled_ledger_survives_window_preparation_as_an_independent_copy(self):
        notes=[midi.Note(0.,1.,60,80,1),midi.Note(2.,3.,64,80,1)]
        score=piano.compile_score(notes)
        plan=piano.prepare_plan(compiled_score=score,start=2.,dur=.2)
        self.assertEqual(plan.action_ledger,score.action_ledger)
        plan.action_ledger['records'][0]['source']['end']=99.
        self.assertEqual(score.action_ledger['records'][0]['source']['end'],1.)


if __name__=='__main__':unittest.main()
