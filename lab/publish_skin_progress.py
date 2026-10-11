"""Update the existing band site with measured rig research and open failures."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
WEB = Path('/home/frosty40/h3/results/band_site/web')
ART = ROOT / 'artifacts/realism'
manifest_path = WEB / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
latest_motion = json.loads((ART / 'skin_motion_exposed_holds_window.json').read_text())
latest_poses = json.loads((ART / 'restore_exposed_holds_window.json').read_text())
progress = json.loads((ROOT / 'lab/realism_results/mechanical_path_20261011.json').read_text())
test_count = progress['test_count']
coupled = progress['completed']['coupled']
if len(latest_poses['records']) != 6 or not all(r['accepted'] for r in latest_poses['records']):
    raise RuntimeError('Publishing text requires the recorded six accepted contact poses')
assert latest_motion['samples'] == 21 and latest_motion['held_score_required_observations'] == 74
section = dict(id='rig-realism', title='Rig realism: rendered skin and motion (in progress)',
    intro='The rendered skin now has an independent geometry evaluator and hard pose constraints. '
          'These are measured short stress probes at hand scale 1. The complete performance, '
          'calibrated hand size, natural form and full body clearance remain unverified.',
    items=[
        dict(kind='image', src='media/rig_realism_held_contact_experiment.png', wide=True,
             title='Earlier: finite held contact improves; connected motion still fails',
             note='At this checkpoint, 97 tests passed. The six exported poses pass actual skin, finite exposed-key contact '
                  'and physical bounds; all 14 requested anchors are verified. Independent held-note '
                  'checks find no missing required holds at those instants. Between them, '
                  f"{latest_motion['affected_samples']}/21 observations collide (maximum {latest_motion['maximum_penetration_mm']:.3f} mm), "
                  f"and {latest_motion['held_score_unsupported_observations']}/74 required held-note observations lose support. "
                  'The earlier loose-footprint window missed 43/74. That window comparison also includes '
                  'numerical conditioning/retention changes, so it does not isolate one cause. '
                  'The frozen-pose pair in the left panel changes only initialization budget. '
                  '<a href="media/rig_realism_held_contact_results.json">All recorded trials and source hashes</a>.'),
        dict(kind='image', src='media/rig_realism_held_contact_R_pose.png', title='Right-hand form review: still overcurled',
             note='Frame 24 after stricter contact restoration. The pose is overcurled and contacts sit '
                  'too close to the key fronts. This is a form failure. Human-calibrated contact placement '
                  'and connected wrist/finger/key paths need improvement.'),
        dict(kind='note', title='Hypothesis and ablation',
             note='Sampled finger tubes miss blended skin and triangle crossings. Compare clipped-point '
                  'witnesses with whole-triangle constraints on the same three adjacent poses, keeping '
                  'anatomy and engineering motion limits. Later independent support checks expose off-key '
                  'false contact acceptance; the latest gate also requires finite exposed tops.'),
        dict(kind='image', src='media/rig_realism_contact_experiment.png', wide=True,
             title='Earlier geometry experiment: exported poses improve; motion still fails',
             note='The static gate improves from 4/6 to 6/6 hand poses. All three corrected instants agree '
                  'with Blender within 0.0012 mm and have zero instrument triangle intersections. '
                  'At 240 Hz, 18/21 interpolated observations collide, up to 5.59 mm. '
                  'These earlier geometry results do not certify supported notes, self/body clearance or continuous time.'),
        dict(kind='image', src='media/rig_realism_contact_pose.png', title='Visual review: geometry clearance is insufficient',
             note='Corrected left hand, frame 24. Free fingers still look too straight and the posture '
                  'looks strained. This is an observed form failure, not an approved performance pose.'),
        dict(kind='table', title='Whole saved probe audit — actual skin versus instrument', rows=[
            ['Probe', 'Frames intersecting instrument', 'Maximum penetration'],
            ['Overlap', '77/86', '7.742 mm'], ['Debussy', '86/86', '8.374 mm'], ['Wide chord', '80/86', '6.957 mm']]),
        dict(kind='note', title='Conclusion and next experiment',
             note='Static finite contact is useful but cannot certify motion. Sustained support now passes '
                  'at the exported poses, but contact loss and clipping remain between them. '
                  'Next: human-calibrated placement, coupled key/contact/wrist trajectory planning, complete '
                  'self/interhand/body geometry, continuous collision checks and reference-based form review. '
                  f'The goal remains active. {test_count} unit tests pass; they are not a performance acceptance gate.'),
        dict(kind='note', title='Research graph and reproducible evidence',
             note='AngelX completed its five-node evidence/counterevidence/constraints/synthesis/reviewer '
                  'graph. Its PASS reviews the proposed protocol, not rig results. The reviewed protocol '
                  'requires paired fixed/movable tests across all 258 frozen inputs; those tests remain pending. '
                  '<a href="media/rig_realism_contact_motion.json">Recorded results and source hashes</a>. '
                  '<a href="https://ipc-sim.github.io/file/IPC-paper-350ppi.pdf">Primary IPC paper</a>.')])
section['items'] = [
    dict(kind='image', src='media/rig_realism_mechanical_path_experiment.png', wide=True,
         title='Latest: fewer sampled collisions; piano motion remains rejected',
         note=f'{test_count} tests pass. Coupled neighboring solves reduce instrument-intersecting observations '
              f"from 18/21 to {coupled['affected_nodes']}/21 and maximum depth from 3.328 to {coupled['maximum_penetration_mm']:.3f} mm. "
              'Required held-pitch observations without support decrease from ten to four. '
              'This exploratory treatment also changes assigned held-key commands, so it does not isolate one cause. '
              'The full candidate is rejected and no failed path is exported. '
              '<a href="media/rig_realism_mechanical_path_results.json">Completed evidence and pending experiment snapshot</a>.'),
    dict(kind='note', title='The piano solids also needed correction',
         note='The original rectangular keys intersect black keys by up to 6.25 mm and the keybed by 8.953 mm '
              'in the tested positions. An opt-in notched-key/bed prototype passes key/key and key/case checks '
              'at rest and frames 23–25, plus 63 independently varied stroke configurations in tests. '
              'Its rendered surface matches the audited solids and hand skin within 0.0013 mm. '
              'Shape alone leaves the hand-motion failures unchanged. The original short hinge is preserved; '
              'a calibrated balance pivot, piano action and force model are still unverified.'),
    dict(kind='note', title='Reviewed experiment and hand-size accounting',
         note='The native AngelX five-node trajectory graph completed and approved the experiment method, '
              'using supplied primary reading notes. Its approval is not a rig pass. A matched frozen-pose '
              'versus movable-interior-pose comparison is running with the same geometry and per-node budgets; '
              'the extra movable pose adds solves, which must remain in the accounting. '
              'Measured hand scale 0.7284 (17.7 cm) is opt-in: the existing planner defaults to scale 1. '
              'Fresh generation at the measured size remains required. '
              '<a href="media/rig_realism_trajectory_review.json">Completed native graph review and receipt identity</a>.'),
    dict(kind='image', src='media/rig_realism_drummer_stroke_control.png', wide=True,
         title='Drummer: stronger accents, softer notes, retained grip',
         note='The controller changes stroke height, contact/rebound timing and dimensionless grip activation '
              'while retaining its finger proxies. New checks cover fingers against the opposite hand and '
              'stick, including the wider tip. At 240 Hz, 10/10 demo hits and 16/16 roll hits pass the proxy '
              'gates, with 6 mm minimum within-hand spacing. All twelve stress patterns remain rejected; '
              'the expanded audit finds interhand finger collisions in eight and finger/opposite-stick '
              'collisions in seven. These are kinematic proxies. Rendered hands/body/costume, shells, '
              'feet/pedals, force dynamics and continuous time remain unverified. '
              '<a href="media/rig_realism_drummer_crosshand_results.json">All frozen stress outcomes</a>.'),
] + section['items']
manifest['sections'] = [section] + [s for s in manifest['sections'] if s['id'] != section['id']]
manifest['updated'] = '2026-10-11 mechanical/path and drummer-grip research; full realism remains incomplete'
manifest['status'].insert(0, f'<b>Rig research:</b> {test_count} tests pass. Coupled sampled motion reduces clipping '
    'from 18 to four observations, but the whole path is rejected. Mechanical key clearances and drummer '
    'cross-hand grip checks are added. Latest evidence below; '
    'older experiment and GPU notes retain their original dated context.')
# Keep one current research status when the publishing command is rerun.
manifest['status'] = [manifest['status'][0]] + [s for s in manifest['status'][1:] if not s.startswith('<b>Rig research:</b>')]
media = WEB / 'media'
media.mkdir(exist_ok=True)
for source, destination in (
        (ART / 'skin_contact_experiment.png', 'rig_realism_contact_experiment.png'),
        (ART / 'sat_window_front/f_00024.png', 'rig_realism_contact_pose.png'),
        (ROOT / 'lab/realism_results/skin_contact_motion_20261011.json', 'rig_realism_contact_motion.json'),
        (ART / 'held_contact_experiment.png', 'rig_realism_held_contact_experiment.png'),
        (ART / 'exposed_holds_R_front/f_00024.png', 'rig_realism_held_contact_R_pose.png'),
        (ROOT / 'lab/realism_results/held_contact_20261011.json', 'rig_realism_held_contact_results.json'),
        (ART / 'mechanical_path_experiment.png', 'rig_realism_mechanical_path_experiment.png'),
        (ROOT / 'lab/realism_results/mechanical_path_20261011.json', 'rig_realism_mechanical_path_results.json'),
        (ROOT / 'lab/realism_results/skin_trajectory_review_result.json', 'rig_realism_trajectory_review.json'),
        (ART / 'drums/crosshand_stroke_control.png', 'rig_realism_drummer_stroke_control.png'),
        (ROOT / 'lab/realism_results/drummer_crosshand_20261011.json', 'rig_realism_drummer_crosshand_results.json')):
    shutil.copy2(source, media / destination)
manifest_path.write_text(json.dumps(manifest, indent=1) + '\n')
print('Updated existing band-site manifest and research media')
