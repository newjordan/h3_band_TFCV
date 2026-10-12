import copy
import unittest
from unittest.mock import patch

import numpy as np

from blockout import midi,piano
from blockout.key_actions import BinaryPedalTimeline,propose_key_plan,physical_notes


def evidence(notes,canonical=None):
    original=canonical if canonical is not None else notes
    return dict(records=[dict(input_id=i,output_id=i,
        input_note=[a.start,a.end,a.pitch,a.velocity,a.track],
        output_note=[b.start,b.end,b.pitch,b.velocity,b.track]) for i,(a,b) in enumerate(zip(original,notes))])


class PhysicalKeyPlanTests(unittest.TestCase):
    def test_exact_unison_retains_voices_and_records_shared_strike_policy(self):
        notes=[midi.Note(.01,1.,60,70,1),midi.Note(.025,2.,60,90,2)]
        notes[1].touch=.8
        canonical=[midi.Note(0.,1.,60,80,1),midi.Note(0.,2.,60,80,2)]
        before=copy.deepcopy([vars(n) for n in notes])
        plan=propose_key_plan(notes,evidence(notes,canonical),{0:0,1:0})
        motion=physical_notes(notes,plan)
        self.assertEqual(len(motion),1)
        self.assertEqual((motion[0].start,motion[0].end,motion[0].velocity,motion[0].touch),(.01,2.,90,.8))
        self.assertEqual(plan['groups'][0]['source_ids'],[0,1])
        self.assertAlmostEqual(plan['groups'][0]['timing_differences'][1]['onset_difference'],.015)
        score=piano.compile_score(notes,key_action_plan=plan)
        self.assertEqual([vars(n) for n in notes],before)
        self.assertEqual(len(score.notes),2)
        self.assertEqual(score.action_ledger['summary']['physical_action_proposals'],1)
        self.assertTrue(score.action_ledger['summary']['source_voices_scheduled_once'])
        self.assertFalse(score.action_ledger['summary']['scheduling_bijection'])
        self.assertEqual(score.action_ledger['records'][1]['source']['start'],.025)

    def test_close_repeated_onsets_different_channels_and_same_track_never_alias(self):
        notes=[midi.Note(0.,1.,60,80,1),midi.Note(.02,1.,60,80,2)]
        cases=[(notes,{0:0,1:0}),
               ([midi.Note(0.,1.,60,80,1),midi.Note(0.,1.,60,80,2)],{0:0,1:1})]
        for canonical,channels in cases:
            plan=propose_key_plan(notes,evidence(notes,canonical),channels)
            self.assertEqual(len(plan['groups']),2)
            score=piano.compile_score(notes,key_action_plan=plan)
            self.assertEqual(score.action_ledger['summary']['missing_source_ids'],[])
        same=[midi.Note(0.,1.,60,80,1),midi.Note(0.,1.,60,80,1)]
        self.assertEqual(len(propose_key_plan(same,evidence(same),{0:0,1:0})['groups']),2)

    def test_wrong_source_missing_duplicate_unknown_and_forged_policy_are_rejected(self):
        notes=[midi.Note(0.,1.,60,80,1),midi.Note(1.,2.,64,80,1)]
        plan=propose_key_plan(notes,evidence(notes),{0:0,1:0})
        bad=[]
        for ids in ([],[0,0],[0,2]):
            value=copy.deepcopy(plan);value['groups'][0]['source_ids']=ids;bad.append(value)
        value=copy.deepcopy(plan);value['groups'][0]['velocity']=90;bad.append(value)
        value=copy.deepcopy(plan);value['groups'][0]['end']=.5;bad.append(value)
        value=copy.deepcopy(plan);value['source_sha256']='wrong';bad.append(value)
        for value in bad:
            with self.subTest(value=value),self.assertRaises(ValueError):physical_notes(notes,value)
        shifted=copy.deepcopy(notes);shifted[0].end=1.1
        with self.assertRaises(ValueError):piano.compile_score(shifted,key_action_plan=plan)
        with self.assertRaises(ValueError):propose_key_plan(notes,evidence(notes),{0:0})

    def test_conditional_release_requires_continuous_sustain_and_offset_match(self):
        notes=[midi.Note(0.,2.,60,80,1),midi.Note(.4,.8,84,80,1)]
        with patch.object(piano,'NO_HAND_CROSS',False),patch.object(piano,'REACH_SPLIT',False):
            baseline=piano.compile_score(notes).action_ledger
        self.assertLess(baseline['records'][0]['scheduled_actions'][0]['end'],2.)
        good=BinaryPedalTimeline([(0.,127),(2.,0)],0.,3.)
        plan=propose_key_plan(notes,evidence(notes),{0:0,1:0},baseline,good,conditional_releases=True)
        self.assertLess(plan['groups'][0]['end'],2.)
        for pedal in (BinaryPedalTimeline([(0.,127),(.7,0),(.8,127),(2.,0)],0.,3.),
                      BinaryPedalTimeline([(0.,127),(2.1,0)],0.,3.),
                      BinaryPedalTimeline([(0.,127),(2.,0)],.5,3.)):
            result=propose_key_plan(notes,evidence(notes),{0:0,1:0},baseline,pedal,conditional_releases=True)
            self.assertEqual(result['groups'][0]['end'],2.)
        forged=copy.deepcopy(plan);forged['pedal_hypothesis']['events']=[(0.,0)]
        with self.assertRaises(ValueError):physical_notes(notes,forged)
        default=propose_key_plan(notes,evidence(notes),{0:0,1:0},baseline,good)
        self.assertEqual(default['groups'][0]['end'],2.)

    def test_one_voice_without_pedal_offset_match_prevents_group_release(self):
        notes=[midi.Note(0.,2.,60,80,1),midi.Note(.01,1.8,60,90,2)]
        canonical=[midi.Note(0.,2.,60,80,1),midi.Note(0.,1.8,60,90,2)]
        baseline=piano.compile_score(notes).action_ledger
        # Supply a short, explicit existing scheduling proposal for the test.
        baseline['records'][0]['scheduled_actions'][0]['end']=.5
        pedal=BinaryPedalTimeline([(0.,127),(2.,0)],0.,3.)
        plan=propose_key_plan(notes,evidence(notes,canonical),{0:0,1:0},baseline,pedal,conditional_releases=True)
        self.assertEqual(plan['groups'][0]['end'],2.)

    def test_strict_hold_conflicts_are_retained_and_reported_at_every_stage(self):
        notes=[midi.Note(0.,2.,60,80,1),midi.Note(.4,.8,84,80,1)]
        plan=propose_key_plan(notes,evidence(notes),{0:0,1:0})
        with patch.object(piano,'NO_HAND_CROSS',False),patch.object(piano,'REACH_SPLIT',False):
            score=piano.compile_score(notes,key_action_plan=plan)
        first=score.action_ledger['records'][0]
        self.assertEqual(first['scheduled_actions'][0]['end'],2.)
        for stage in ('key_action_plan','slicing','fingering_repair','finger_reuse','reach_release'):
            self.assertEqual(first['stages'][stage][0]['end'],2.)
        # Forced same-finger holds: event construction cannot silently trim them.
        for i,n in enumerate(notes):n._score_event_id=i
        fing=[[(notes[0],2)],[(notes[1],2)]]
        events=piano.finger_events(fing,preserve_holds=True)
        self.assertEqual(events[2][0][1],2.)
        from blockout.key_actions import event_stage,interval_conflicts
        conflicts=interval_conflicts({'R':{'action_trace':{'reach_release':event_stage(fing,events)}}})
        self.assertEqual(len(conflicts),1)
        self.assertFalse(conflicts[0]['same_pitch'])

    def test_long_requested_hold_survives_event_selection_and_copy_metadata(self):
        notes=[midi.Note(0.,20.,60,80,1)]
        plan=propose_key_plan(notes,evidence(notes),{0:0})
        score=piano.compile_score(notes,key_action_plan=plan)
        self.assertEqual(len(piano._event_candidates(score.hands['R'],0,12.)),1)
        legacy=piano.compile_score(notes)
        self.assertEqual(piano._event_candidates(legacy.hands['R'],0,12.),[])
        prepared=piano.prepare_plan(compiled_score=score,start=12.,dur=.2)
        self.assertEqual(prepared.physical_key_plan,score.physical_key_plan)
        self.assertTrue(np.all(prepared.hands['R']['activation'][:,0]>.7))
        prepared.physical_key_plan['groups'][0]['end']=99.
        self.assertEqual(score.physical_key_plan['groups'][0]['end'],20.)

    def test_plan_respects_speed_clock_without_changing_source(self):
        notes=[midi.Note(2.,4.,60,80,1)]
        plan=propose_key_plan(notes,evidence(notes),{0:0})
        score=piano.compile_score(notes,speed=2.,key_action_plan=plan)
        self.assertEqual(score.notes[0].start,1.)
        self.assertEqual(score.hands['R']['ev'][0][0][:2],[1.,2.])
        self.assertEqual((notes[0].start,notes[0].end),(2.,4.))
        self.assertEqual(score.physical_key_plan['clock_speed'],2.)


if __name__=='__main__':unittest.main()
