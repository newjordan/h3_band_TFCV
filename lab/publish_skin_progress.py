"""Update the existing band site with measured rig research and open failures."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
WEB = Path('/home/frosty40/h3/results/band_site/web')
ART = ROOT / 'artifacts/realism'
manifest_path = WEB / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
section = dict(id='rig-realism', title='Rig realism: rendered skin and motion (in progress)',
    intro='The rendered skin now has an independent geometry evaluator and hard pose constraints. '
          'These are measured short stress probes at hand scale 1. The complete performance, '
          'calibrated hand size, natural form and full body clearance remain unverified.',
    items=[
        dict(kind='note', title='Hypothesis and ablation',
             note='Sampled finger tubes miss blended skin and triangle crossings. Compare clipped-point '
                  'witnesses with whole-triangle constraints on the same three adjacent poses, keeping '
                  'anatomy and engineering motion limits. Actual skin anchors preserve existing supported notes.'),
        dict(kind='image', src='media/rig_realism_contact_experiment.png', wide=True,
             title='Result: exported poses improve; motion still fails',
             note='The static gate improves from 4/6 to 6/6 hand poses. All three corrected instants agree '
                  'with Blender within 0.0012 mm and have zero instrument triangle intersections. '
                  'At 240 Hz, 18/21 interpolated observations collide, up to 5.59 mm. '
                  'The gate does not certify musical completeness, self/body clearance or continuous time.'),
        dict(kind='image', src='media/rig_realism_contact_pose.png', title='Visual review: geometry clearance is insufficient',
             note='Corrected left hand, frame 24. Free fingers still look too straight and the posture '
                  'looks strained. This is an observed form failure, not an approved performance pose.'),
        dict(kind='table', title='Whole saved probe audit — actual skin versus instrument', rows=[
            ['Probe', 'Frames intersecting instrument', 'Maximum penetration'],
            ['Overlap', '77/86', '7.742 mm'], ['Debussy', '86/86', '8.374 mm'], ['Wide chord', '80/86', '6.957 mm']]),
        dict(kind='note', title='Conclusion and next experiment',
             note='Static feasibility is useful but cannot certify motion. A key begins returning while '
                  'its score note is held; requiring the missing sustained contact exposes a rejected '
                  'solver failure. Next: coupled key/contact/wrist trajectory planning, complete '
                  'self/interhand/body geometry, continuous collision checks and reference-based form review. '
                  'The goal remains active. 92 unit tests pass; they are not a performance acceptance gate.'),
        dict(kind='note', title='Research graph and reproducible evidence',
             note='AngelX completed its five-node evidence/counterevidence/constraints/synthesis/reviewer '
                  'graph. Its PASS reviews the proposed protocol, not rig results. The reviewed protocol '
                  'requires paired fixed/movable tests across all 258 frozen inputs; those tests remain pending. '
                  '<a href="media/rig_realism_contact_motion.json">Recorded results and source hashes</a>. '
                  '<a href="https://ipc-sim.github.io/file/IPC-paper-350ppi.pdf">Primary IPC paper</a>.')])
manifest['sections'] = [section] + [s for s in manifest['sections'] if s['id'] != section['id']]
manifest['updated'] = '2026-10-11 rendered-skin/contact research; full realism remains incomplete'
manifest['status'].insert(0, '<b>Rig research:</b> 92 tests pass. Actual skin constraints clear six '
    'adjacent exported hand poses, but interpolated motion still clips. Latest evidence below; '
    'older experiment and GPU notes retain their original dated context.')
# Keep one current research status when the publishing command is rerun.
manifest['status'] = [manifest['status'][0]] + [s for s in manifest['status'][1:] if not s.startswith('<b>Rig research:</b>')]
media = WEB / 'media'
media.mkdir(exist_ok=True)
for source, destination in (
        (ART / 'skin_contact_experiment.png', 'rig_realism_contact_experiment.png'),
        (ART / 'sat_window_front/f_00024.png', 'rig_realism_contact_pose.png'),
        (ROOT / 'lab/realism_results/skin_contact_motion_20261011.json', 'rig_realism_contact_motion.json')):
    shutil.copy2(source, media / destination)
manifest_path.write_text(json.dumps(manifest, indent=1) + '\n')
print('Updated existing band-site manifest and research media')
