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


def note_stage(notes):
    result = []
    for n in notes:
        if (not isinstance(n._score_event_id, Integral) or n._score_event_id<0
                or not math.isfinite(n.start) or not math.isfinite(n.end) or n.end<=n.start
                or any(not isinstance(v, Integral) for v in (n.pitch,n.velocity,n.track))):
            raise ValueError('Source attacks require finite positive duration and exact integer identities')
        result.append(dict(source_id=int(n._score_event_id), start=float(n.start), end=float(n.end),
                           pitch=int(n.pitch), velocity=int(n.velocity), track=int(n.track)))
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
                identity = record['source_id']
                if identity not in rows:
                    raise ValueError('Action lineage contains an unknown source attack')
                row = rows[identity]
                if row['hand'] not in (None, hand):
                    raise ValueError('Source attack assigned to both hands')
                row['hand'] = hand
                row['stages'].setdefault(stage, []).append(record)
    missing, duplicates, shifted, shortened, extended = [], [], [], [], []
    for identity, row in rows.items():
        previous = [row['source']]
        for stage in ('slicing', 'fingering_repair', 'finger_reuse', 'reach_release'):
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
    return dict(schema='piano-source-action-ledger/v1', input_scope=scope,
        source_sha256=hashlib.sha256(ledger_bytes).hexdigest(), records=list(rows.values()),
        summary=dict(source_attacks=len(source), scheduled_actions=sum(len(r['scheduled_actions']) for r in rows.values()),
            missing_source_ids=missing, duplicate_source_ids=duplicates,
            shifted_source_ids=sorted(set(shifted)), shortened_source_ids=sorted(set(shortened)),
            extended_source_ids=sorted(set(extended)), all_attacks_accounted=len(rows)==len(source),
            scheduling_bijection=not missing and not duplicates),
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
