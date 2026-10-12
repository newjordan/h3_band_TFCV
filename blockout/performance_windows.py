"""Render partitions of a solved trajectory, without starting a new simulation."""
import copy
import hashlib
import json


def _digest(frames):
    return hashlib.sha256(json.dumps(frames, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def export_window(animation, first, last):
    """Select [first,last); retain the source clock for exact, checked rejoining."""
    count = len(animation['frames'])
    if not isinstance(first, int) or not isinstance(last, int) or not 0 <= first < last <= count:
        raise ValueError('A nonempty in-range integer frame interval is required')
    if 'trajectory_partition' in animation:
        raise ValueError('Partition the original solved trajectory directly')
    result = copy.deepcopy(animation)
    source_frames = animation['frames'][first:last]
    source_times = [f['t'] for f in source_frames]
    offset = source_times[0]
    result['start'] = animation['start'] + offset
    result['frames'] = copy.deepcopy(source_frames)
    for frame in result['frames']:
        frame['t'] -= offset
    result['warmup'] = {'scope': 'retained in the original solved trajectory'}
    result['qa'] = {'scope': 'full solve diagnostics retained in the original trajectory; recompute for this window'}
    for key in ('beats', 'downbeats'):
        result[key] = [t-offset for t in animation.get(key, [])
                       if offset <= t < offset + len(source_frames)/animation['fps']]
    result['trajectory_partition'] = dict(source_start=animation['start'], first=first, last=last,
        source_count=count, source_times=source_times, source_sha256=_digest(animation['frames']),
        partition_sha256=_digest(source_frames), scope='selected solved frames; only the time origin changes')
    return result


def rejoin_windows(parts):
    """Reject gaps, reordered/tampered poses, mixed sources or incomplete coverage."""
    if not parts:
        raise ValueError('At least one partition is required')
    first = parts[0]['trajectory_partition']
    frames, next_index = [], 0
    for part in parts:
        meta = part['trajectory_partition']
        if (meta['source_sha256'] != first['source_sha256'] or
                meta['source_start'] != first['source_start'] or
                meta['source_count'] != first['source_count'] or part['fps'] != parts[0]['fps'] or
                meta['first'] != next_index or meta['last']-meta['first'] != len(part['frames']) or
                len(meta['source_times']) != len(part['frames'])):
            raise ValueError('Partitions must cover one trajectory in order without gaps')
        restored = copy.deepcopy(part['frames'])
        offset = meta['source_times'][0]
        if part['start'] != meta['source_start'] + offset:
            raise ValueError('Partition time origin changed')
        for frame, t in zip(restored, meta['source_times']):
            if frame['t'] != t-offset:
                raise ValueError('Partition timestamps changed')
            frame['t'] = t
        if _digest(restored) != meta['partition_sha256']:
            raise ValueError('Partition pose or key data changed')
        frames.extend(restored)
        next_index = meta['last']
    if next_index != first['source_count'] or _digest(frames) != first['source_sha256']:
        raise ValueError('Partitions do not reconstruct the complete source trajectory')
    result = copy.deepcopy(parts[0])
    result.pop('trajectory_partition')
    result['start'], result['frames'] = first['source_start'], frames
    result['qa'] = {'scope': 'reconstructed identical source frames; full solve diagnostics belong to the source'}
    return result
