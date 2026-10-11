"""Publish the verified continuous performance preview with its actual scope."""


def apply_performance_preview(section, report):
    assert report['status'] == 'complete' and not report['full_performance_accepted']
    section['items'].insert(0, dict(kind='video', src='media/rig_realism_piano_60s.mp4', wide=True,
        title='One minute of piano performance, with audio',
        note='Fresh grayscale render at 768×432 and 12 fps, with a fixed view of both hands and keys. '
             'The continuous passage covers 239.9–299.9 seconds of the original score; playback tempo '
             'and audio timing are preserved. This is the earlier complete v19 planner using current '
             'skinned geometry, with its original hand size 1. The latest measured-size static contact '
             'experiments below have not passed and are not integrated into this motion. Curling, '
             'spacing and clipping remain visible for review; no realistic full performance is accepted. '
             '<a href="media/rig_realism_performance_preview_results.json">Source, timing, camera and render identity</a>.'))
