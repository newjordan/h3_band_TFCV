"""Separate source sound events, proposed finger actions and estimated pedals.

This records scheduling changes and ideal binary sustain hypotheses. It does
not identify a unique physical key release, establish pedal ground truth, or
certify actual key/finger/foot contacts, forces, naturalness or geometry.
"""
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from numbers import Integral
import copy


def source_records(notes):
    """Stable source-order identity; dynamic motion attributes are not sound events."""
    if not notes or any(not math.isfinite(n.start) or not math.isfinite(n.end) or n.end<=n.start
            or any(not isinstance(v,Integral) for v in (n.pitch,n.velocity,n.track)) for n in notes):
        raise ValueError('Finite source gates and exact integer identities are required')
    return [dict(source_id=i, start=float(n.start), end=float(n.end), pitch=int(n.pitch),
                 velocity=int(n.velocity), track=int(n.track)) for i, n in enumerate(notes)]


def source_digest(notes):
    return hashlib.sha256(json.dumps(source_records(notes), separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def propose_key_plan(notes, timing_receipt, source_channels, baseline_ledger=None,
                     pedal=None, *, conditional_releases=False, offset_tolerance=.030):
    """Experimental physical actions, retaining every original source voice.

    Channel identity must come from the caller's verified input MIDI evidence.
    Exact canonical unisons may share a strike; nearby repeated onsets may not.
    An estimated pedal can support a release hypothesis, never identify one.
    """
    tolerance = float(offset_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('A finite nonnegative offset comparison window is required')
    records = timing_receipt['records']
    count = len(notes)
    if (len(records) != count or sorted(r['input_id'] for r in records) != list(range(count))
            or sorted(r['output_id'] for r in records) != list(range(count))
            or set(source_channels) != set(range(count))):
        raise ValueError('Complete one-to-one timing and channel evidence is required')
    by_output = {r['output_id']: r for r in records}
    bases, buckets = {}, {}
    for i, n in enumerate(notes):
        r = by_output[i]
        if r['output_note'] != [n.start,n.end,n.pitch,n.velocity,n.track]:
            raise ValueError('Timing evidence does not match this original source')
        ch = source_channels[r['input_id']]
        if not isinstance(ch, Integral) or not 0 <= ch <= 15:
            raise ValueError('Explicit integer MIDI channel evidence is required')
        bases[i] = dict(input_id=r['input_id'], input_note=copy.deepcopy(r['input_note']),
                        channel=int(ch))
        key = (r['input_note'][0],r['input_note'][2],int(ch))
        buckets.setdefault(key, []).append(i)
    partitions = []
    for ids in buckets.values():
        # Multiple same-track attacks are ambiguous; keep the whole bucket separate.
        tracks = [bases[i]['input_note'][4] for i in ids]
        partitions.extend([ids] if len(set(tracks)) == len(tracks) else [[i] for i in ids])
    baseline = {}
    if baseline_ledger is not None:
        if baseline_ledger['source_sha256'] != source_digest(notes):
            raise ValueError('Release proposals belong to a different original source')
        baseline = {r['source']['source_id']:r['scheduled_actions'] for r in baseline_ledger['records']}
    groups = []
    for ids in sorted(partitions, key=min):
        ids = sorted(ids)
        strongest = max(ids,key=lambda i:(notes[i].velocity,-notes[i].start,-i))
        onset, conservative = min(notes[i].start for i in ids), max(notes[i].end for i in ids)
        end, evidence = conservative, []
        proposals = [a['end'] for i in ids for a in baseline.get(i,[])]
        candidate = max(proposals) if proposals else None
        if conditional_releases and pedal is not None and candidate is not None and onset < candidate < conservative:
            for i in ids:
                n = notes[i]
                if not pedal.start <= n.start <= candidate <= n.end <= pedal.end:
                    evidence.append(dict(source_id=i,status='outside domain or release outside source gate',accepted=False))
                    continue
                coverage = pedal.release_coverage(candidate,n.end)
                hypotheses = pedal.release_hypotheses(n.start,n.end,tolerance)
                evidence.append(dict(source_id=i,coverage=coverage,offset_hypotheses=hypotheses,
                    accepted=coverage['covered'] and bool(hypotheses)))
            if evidence and all(r['accepted'] for r in evidence):
                end = candidate
        groups.append(dict(group_id=min(ids),source_ids=ids,start=float(onset),end=float(end),
            conservative_end=float(conservative),pitch=int(notes[ids[0]].pitch),
            velocity=int(notes[strongest].velocity),track=int(notes[strongest].track),
            attribute_source_id=strongest,shared_strike=len(ids)>1,
            timing_differences=[dict(source_id=i,onset_difference=float(notes[i].start-onset),
                release_difference=float(notes[i].end-end),velocity_difference=int(notes[i].velocity-notes[strongest].velocity)) for i in ids],
            release_candidate=candidate,release_evidence=evidence,
            release_policy='estimated pedal hypothesis' if end<conservative else 'conservative source gate',
            canonical_evidence={str(i):bases[i] for i in ids}))
    plan = dict(schema='piano-physical-key-plan/v1',source_sha256=source_digest(notes),groups=groups,
        conditional_releases=bool(conditional_releases),offset_tolerance=tolerance,
        pedal_hypothesis=dict(events=list(pedal.events),start=pedal.start,end=pedal.end,
                              initial_value=pedal.initial_value) if pedal is not None else None,
        physical_releases_validated=False,ground_truth=False,waveform_equivalence_verified=False,
        original_soundtrack_replaced=False,full_performance_accepted=False)
    physical_notes(notes,plan)  # Validate the complete partition before it can reach the controller.
    return plan


def physical_notes(notes, plan):
    """Validate a plan and copy its motion notes; the input ledger stays intact."""
    if plan.get('schema') != 'piano-physical-key-plan/v1' or plan.get('source_sha256') != source_digest(notes):
        raise ValueError('Physical key plan does not identify this original source')
    members = [i for g in plan['groups'] for i in g['source_ids']]
    if any(not isinstance(i, Integral) for i in members) or sorted(members) != list(range(len(notes))):
        raise ValueError('Physical actions require a complete disjoint source partition')
    input_ids=[g['canonical_evidence'][str(i)]['input_id'] for g in plan['groups'] for i in g['source_ids']]
    if sorted(input_ids)!=list(range(len(notes))):
        raise ValueError('Canonical source evidence requires a complete input partition')
    result, group_ids = [], set()
    for g in plan['groups']:
        ids = g['source_ids']
        if not ids or g['group_id'] != min(ids) or g['group_id'] in group_ids:
            raise ValueError('Every physical action requires a unique stable group identity')
        group_ids.add(g['group_id'])
        values = [g['canonical_evidence'][str(i)] for i in ids]
        keys = {(v['input_note'][0],v['input_note'][2],v['channel']) for v in values}
        if (len(keys)!=1 or any(not isinstance(v['channel'],Integral) or not 0<=v['channel']<=15 for v in values)
                or any(v['input_note'][2]!=notes[i].pitch or v['input_note'][4]!=notes[i].track for i,v in zip(ids,values))
                or (len(ids)>1 and len({v['input_note'][4] for v in values})!=len(ids))):
            raise ValueError('Only exact proven canonical unisons on one channel may share a strike')
        strongest = max(ids,key=lambda i:(notes[i].velocity,-notes[i].start,-i))
        if (g['attribute_source_id']!=strongest or g['start']!=min(notes[i].start for i in ids)
                or g['conservative_end']!=max(notes[i].end for i in ids)
                or any(notes[i].pitch!=g['pitch'] for i in ids)
                or g['velocity']!=notes[strongest].velocity or g['track']!=notes[strongest].track
                or not math.isfinite(g['end']) or not g['start']<g['end']<=g['conservative_end']):
            raise ValueError('Physical action policy or source identity changed')
        if g['end'] < g['conservative_end'] and (not plan['conditional_releases']
                or g['end']!=g['release_candidate'] or len(g['release_evidence'])!=len(ids)
                or {r['source_id'] for r in g['release_evidence']}!=set(ids)
                or not all(r['accepted'] for r in g['release_evidence'])):
            raise ValueError('Early releases require explicit evidence for every represented voice')
        if g['end'] < g['conservative_end']:
            if plan.get('pedal_hypothesis') is None:
                raise ValueError('An early release requires its bounded pedal hypothesis')
            timeline=BinaryPedalTimeline(**plan['pedal_hypothesis'])
            for i in ids:
                if (not timeline.start<=notes[i].start<=g['end']<=notes[i].end<=timeline.end
                        or not timeline.release_coverage(g['end'],notes[i].end)['covered']
                        or not timeline.release_hypotheses(notes[i].start,notes[i].end,plan['offset_tolerance'])):
                    raise ValueError('Recorded early release does not satisfy its pedal hypothesis')
        n = copy.copy(notes[strongest])
        n.start,n.end,n.velocity,n.track=g['start'],g['end'],g['velocity'],g['track']
        n._score_event_id=min(ids)
        n._action_source_ids=tuple(ids)
        n._physical_group_id=g['group_id']
        result.append(n)
    return sorted(result,key=lambda n:(n.start,n.pitch,n._score_event_id))


def interval_conflicts(hands):
    """Requested intervals sharing one finger, before any geometry solve."""
    result=[]
    for side,h in hands.items():
        records=h['action_trace']['reach_release']
        for finger in range(5):
            active=[]
            for r in sorted((r for r in records if r['finger']==finger),key=lambda r:r['start']):
                active=[a for a in active if a['end']>r['start']]
                for a in active:
                    result.append(dict(hand=side,finger=finger,first_group=a['source_id'],
                        second_group=r['source_id'],start=r['start'],end=min(a['end'],r['end']),
                        same_pitch=a['pitch']==r['pitch'],first_pitch=a['pitch'],second_pitch=r['pitch']))
                active.append(r)
    return result


def note_stage(notes):
    result = []
    for n in notes:
        if (not isinstance(n._score_event_id, Integral) or n._score_event_id<0
                or not math.isfinite(n.start) or not math.isfinite(n.end) or n.end<=n.start
                or any(not isinstance(v, Integral) for v in (n.pitch,n.velocity,n.track))):
            raise ValueError('Source attacks require finite positive duration and exact integer identities')
        record=dict(source_id=int(n._score_event_id), start=float(n.start), end=float(n.end),
                    pitch=int(n.pitch), velocity=int(n.velocity), track=int(n.track))
        if hasattr(n,'_action_source_ids'):
            record.update(represented_source_ids=list(n._action_source_ids),physical_group_id=n._physical_group_id)
        result.append(record)
    return result


def event_stage(fingering, events):
    """Associate legacy seven-value events with source IDs without changing them."""
    result = []
    for finger in range(5):
        notes = sorted([n for row in fingering for n, f in row if f == finger], key=lambda n: n.start)
        if len(notes) != len(events[finger]):
            raise ValueError('Scheduled event count does not match its source lineage')
        for note, event in zip(notes, events[finger]):
            if (event[2], event[3]) != (note.pitch, note.velocity):
                raise ValueError('Scheduled event identity changed')
            record = note_stage([note])[0]
            record.update(start=float(event[0]), end=float(event[1]), finger=finger)
            result.append(record)
    return result


def action_ledger(notes, hands, scope='complete original input'):
    """Keep missing/duplicate/shifted source attacks visible at every stage."""
    source = note_stage(notes)
    identities = [r['source_id'] for r in source]
    if len(set(identities)) != len(identities):
        raise ValueError('Every source attack requires one unique source ID')
    originals = {r['source_id']: r for r in source}
    rows = {i: dict(source=originals[i], hand=None, stages={}, changes=[]) for i in identities}
    for hand, value in hands.items():
        for stage, records in value['action_trace'].items():
            for record in records:
                for identity in record.get('represented_source_ids',[record['source_id']]):
                    if identity not in rows:
                        raise ValueError('Action lineage contains an unknown source attack')
                    row = rows[identity]
                    if row['hand'] not in (None, hand):
                        raise ValueError('Source attack assigned to both hands')
                    row['hand'] = hand
                    row['stages'].setdefault(stage, []).append(copy.deepcopy(record))
    missing, duplicates, shifted, shortened, extended = [], [], [], [], []
    for identity, row in rows.items():
        previous = [row['source']]
        stages=('key_action_plan',) if 'key_action_plan' in row['stages'] else ()
        for stage in stages+('slicing', 'fingering_repair', 'finger_reuse', 'reach_release'):
            current = row['stages'].get(stage, [])
            if current != previous:
                row['changes'].append(dict(stage=stage, before=previous, after=current))
            previous = current
        scheduled = row['stages'].get('reach_release', [])
        row['scheduled_actions'] = scheduled
        if not scheduled:
            missing.append(identity)
        if len(scheduled) > 1:
            duplicates.append(identity)
        for record in scheduled:
            if record['start'] != row['source']['start']:
                shifted.append(identity)
            if record['end'] < row['source']['end']:
                shortened.append(identity)
            elif record['end'] > row['source']['end']:
                extended.append(identity)
    ledger_bytes = json.dumps(source, separators=(',', ':'), allow_nan=False).encode()
    physical_count=sum(len(h['action_trace']['reach_release']) for h in hands.values())
    shared=any(len(r.get('represented_source_ids',[]))>1 for h in hands.values() for r in h['action_trace']['reach_release'])
    return dict(schema='piano-source-action-ledger/v1', input_scope=scope,
        source_sha256=hashlib.sha256(ledger_bytes).hexdigest(), records=list(rows.values()),
        summary=dict(source_attacks=len(source), scheduled_actions=sum(len(r['scheduled_actions']) for r in rows.values()),
            missing_source_ids=missing, duplicate_source_ids=duplicates,
            shifted_source_ids=sorted(set(shifted)), shortened_source_ids=sorted(set(shortened)),
            extended_source_ids=sorted(set(extended)), all_attacks_accounted=len(rows)==len(source),
            scheduling_bijection=not missing and not duplicates and not shared,
            source_voices_scheduled_once=not missing and not duplicates,
            physical_action_proposals=physical_count,shared_physical_actions=shared),
        physical_releases_validated=False, pedal_coverage_validated=False,
        full_performance_accepted=False,
        scope='Scheduling proposals with immutable source timing; actual pressing and sustain are unverified')


class BinaryPedalTimeline:
    """Explicit bounded MIDI CC64 hypothesis with inherited crop state.

    Retain original events before and after the query domain. Values <=63 are
    up and >=64 down. Contradictory coincident transitions are ambiguous and
    rejected. Binary states do not encode foot position or half-pedal physics.
    """
    def __init__(self, events, start, end, initial_value=0):
        self.start, self.end = float(start), float(end)
        if not math.isfinite(self.start) or not math.isfinite(self.end) or self.end <= self.start:
            raise ValueError('A finite positive timeline domain is required')
        if not isinstance(initial_value, Integral) or not 0 <= initial_value <= 127:
            raise ValueError('Initial MIDI value must be from 0 to 127')
        self.initial_value = int(initial_value)
        events = list(events)
        self.events = tuple((float(t), int(v)) for t, v in events)
        # Check original values before int conversion so fractions are not hidden.
        for (_, value), (time, _) in zip(events, self.events):
            if not math.isfinite(time) or not isinstance(value, Integral) or not 0 <= value <= 127:
                raise ValueError('Finite times and integer MIDI values from 0 to 127 are required')
        for before, after in zip(self.events, self.events[1:]):
            if after[0] < before[0] or (after[0] == before[0] and (after[1]>=64) != (before[1]>=64)):
                raise ValueError('Pedal transitions must be ordered and unambiguous at coincident times')
        self.times = tuple(t for t, _ in self.events)

    def _time(self, value):
        value = float(value)
        if not math.isfinite(value) or not self.start <= value <= self.end:
            raise ValueError('Query outside the known pedal hypothesis domain')
        return value

    def state(self, time):
        time = self._time(time)
        index = bisect_right(self.times, time)-1
        return (self.events[index][1] if index >= 0 else self.initial_value) >= 64

    def release_coverage(self, release, audible_offset):
        """Does one proposed early release have uninterrupted ideal sustain?"""
        a, b = self._time(release), self._time(audible_offset)
        if b < a:
            raise ValueError('Audible offset must not precede key release')
        interruptions = [t for t, v in self.events[bisect_right(self.times,a):bisect_left(self.times,b)] if v < 64]
        covered = a == b or (self.state(a) and not interruptions)
        return dict(covered=covered, shortened=a<b, state_at_release=self.state(a),
            intervening_up_times=interruptions, release=a, audible_offset=b,
            scope='Ideal binary sustain hypothesis; not an observed physical release or acoustic equivalence')

    def release_hypotheses(self, onset, audible_offset, offset_tolerance=0.):
        """Return nonunique intervals for offsets near estimated pedal releases.

        This is an inverse ambiguity diagnostic. It does not select a key-off,
        omit an attack, or include hammer/key travel requirements. Tolerance
        is an explicit comparison window, not model timing accuracy.
        """
        onset, offset = self._time(onset), self._time(audible_offset)
        tolerance = float(offset_tolerance)
        if offset < onset or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError('Ordered event times and finite nonnegative tolerance are required')
        down = self.start if self.initial_value>=64 else None
        result = []
        for time, value in self.events:
            if value>=64:
                if down is None:
                    down = time
            elif down is not None:
                if abs(time-offset)<=tolerance and max(onset,down,self.start)<min(time,self.end):
                    result.append(dict(earliest=max(onset,down,self.start), latest=min(time,self.end),
                        pedal_down=down, pedal_up=time, offset_difference=time-offset,
                        offset_tolerance=tolerance, unique=False))
                down = None
        return result


def legacy_sampler_damping(release, reset_times):
    """Exact damping rule of the existing sampler, not a measured pedal model."""
    release = float(release)
    resets = tuple(float(t) for t in reset_times)
    if not math.isfinite(release) or not all(math.isfinite(t) for t in resets) or any(b<a for a,b in zip(resets,resets[1:])):
        raise ValueError('Finite release and ordered finite reset times are required')
    index = bisect_left(resets, release)
    return max(release, resets[index]+.04) if index < len(resets) else release+3.
