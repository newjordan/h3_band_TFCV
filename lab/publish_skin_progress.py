"""Update the existing band site with measured rig research and open failures."""
import json
from pathlib import Path
import shutil
from mesh_progress_site import apply_latest
from deformation_progress_site import apply_deformation
from performance_preview_site import apply_performance_preview
from composure_site import apply_composure
from hand_order_site import apply_hand_order
from continuous_plan_site import apply_continuous_plan, apply_full_suite
from orientation_site import apply_orientation
from key_action_site import apply_key_actions

ROOT = Path(__file__).resolve().parents[1]
WEB = Path('/home/frosty40/h3/results/band_site/web')
ART = ROOT / 'artifacts/realism'
manifest_path = WEB / 'manifest.json'
manifest = json.loads(manifest_path.read_text())
latest_motion = json.loads((ART / 'skin_motion_exposed_holds_window.json').read_text())
latest_poses = json.loads((ART / 'restore_exposed_holds_window.json').read_text())
progress = json.loads((ROOT / 'lab/realism_results/mechanical_path_20261011.json').read_text())
human = json.loads((ROOT / 'lab/realism_results/human_contact_pilot_20261011.json').read_text())
mesh = json.loads((ROOT / 'lab/realism_results/mesh_waypoint_20261011.json').read_text())
deformation = json.loads((ROOT / 'lab/realism_results/deformation_and_union_20261011.json').read_text())
preview = json.loads((ROOT / 'lab/realism_results/performance_preview_20261011.json').read_text())
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
    dict(kind='image', src='media/rig_realism_human_contact_pilot.png', wide=True,
         title='Latest measured-size pilot: posture improves locally; deeper contact still fails',
         note=f'{test_count} tests pass. Fresh generation uses hand scale 0.7284. The complete saved '
              '86-frame probe intersects the instrument in 63 frames (maximum 5.909 mm); within the '
              '49-frame main interval, 36 intersect. The matched four-treatment R-frame-24 pilot keeps '
              'the same anatomy, actual skin, held notes, neighboring poses, speed/spacing gates and solver budgets. '
              'Contact-only and posture-guided corrections pass their one-pose gates and retain four actual '
              'skin anchors. Guidance reduces the soft reference deviation from 7.24 to 3.74. '
              'Both calibrated-depth treatments fail, including 2.170 mm penetration and four unsupported '
              'anchors, and no rejected pose is exported. Broader wrist/contact planning is still needed. '
              '<a href="media/rig_realism_human_contact_results.json">All four outcomes, hashes and baseline intervals</a>.'),
    dict(kind='image', src='media/rig_realism_measured_contact_only.png',
         title='Measured-size contact correction: fixed camera baseline',
         note='One accepted right-hand pose at frame 24. Several contacts sit near the key fronts and the '
              'hand remains overcurled. The other hand and the connected path are unvalidated.'),
    dict(kind='image', src='media/rig_realism_measured_form_guidance.png',
         title='Same camera, added soft posture guidance: naturalness remains unresolved',
         note='The four held contacts and hard local gates remain satisfied, with 0.466 mm minimum proxy '
              'finger clearance. Actual landmark angles guide fingers 2–5 using pooled free/pressing '
              'reference bands. These are soft statistics, not muscle activation or a human acceptance test. '
              'The hand still looks overcurled. Both hands, topology and key solids agree with Blender '
              'within 0.01 mm in the independent measured-size parity check. '
              'New exports preserve solver precision and explicit hand size; conflicting size overrides are rejected. '
              '<a href="https://arxiv.org/html/2410.05791v1">FürElise primary paper</a>.'),
    dict(kind='image', src='media/rig_realism_mechanical_path_experiment.png', wide=True,
         title='Latest: fewer sampled collisions; piano motion remains rejected',
         note=f'{test_count} tests pass. Coupled neighboring solves reduce instrument-intersecting observations '
              f"from 18/21 to {coupled['affected_nodes']}/21 and maximum depth from 3.328 to {coupled['maximum_penetration_mm']:.3f} mm. "
              'Required held-pitch observations without support decrease from ten to four. '
              'This exploratory treatment also changes assigned held-key commands, so it does not isolate one cause. '
              'The full candidate is rejected and no failed path is exported. '
              '<a href="media/rig_realism_mechanical_path_results.json">Completed evidence and matched trajectory results</a>.'),
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
              'versus movable-interior-pose comparison has completed: both paths are rejected, retaining '
              'four clipped nodes and four unsupported held-pitch observations. Their per-node budgets '
              'match; the frozen path uses 72 local solves and the movable path 76. '
              'Measured hand scale 0.7284 (17.7 cm) is opt-in: the existing planner defaults to scale 1. '
              'The separate fresh measured-size baseline and posture pilot above preserve that size. '
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
apply_latest(section, mesh)
apply_deformation(section, deformation)
apply_performance_preview(section, preview)
composure_path = ROOT / 'lab/realism_results/composure_20261011.json'
composure = json.loads(composure_path.read_text()) if composure_path.exists() else None
if composure is not None and composure['status'] == 'complete':
    apply_composure(section, composure)
hand_order_path = ROOT / 'lab/realism_results/hand_order_20261011.json'
hand_order = json.loads(hand_order_path.read_text()) if hand_order_path.exists() else None
if hand_order is not None and hand_order['status'] == 'complete':
    apply_hand_order(section, hand_order)
continuous_plan_path = ROOT / 'lab/realism_results/continuous_plan_20261011.json'
continuous_plan = json.loads(continuous_plan_path.read_text()) if continuous_plan_path.exists() else None
if continuous_plan is not None and continuous_plan['status'] == 'complete':
    apply_continuous_plan(section, continuous_plan)
orientation_path = ROOT / 'lab/realism_results/orientation_20261011.json'
orientation = json.loads(orientation_path.read_text()) if orientation_path.exists() else None
if orientation is not None and orientation['status'] == 'complete':
    apply_orientation(section, orientation)
key_action_path = ROOT / 'lab/realism_results/key_actions_20261011.json'
key_action = json.loads(key_action_path.read_text()) if key_action_path.exists() else None
if key_action is not None and key_action['status'] == 'complete':
    apply_key_actions(section, key_action)
full_suite_path = ROOT / 'lab/realism_results/full_suite_progress_20261011.json'
full_suite = json.loads(full_suite_path.read_text()) if full_suite_path.exists() else None
if full_suite is not None and full_suite['status'] == 'complete':
    apply_full_suite(section, full_suite)
manifest['sections'] = [section] + [s for s in manifest['sections'] if s['id'] != section['id']]
manifest['updated'] = '2026-10-11 one-minute grayscale piano performance with audio, frozen deformation study and connected key-top pilot; complete realism remains unfinished'
manifest['status'].insert(0, f"<b>Rig research:</b> {deformation['unit_tests']['passed']} tests pass. "
    'Identical-bone quaternion deformation leaves playing poses rejected. Uncurling removes self folds '
    'but clips the keys. Connected physical key-top contact is now an opt-in planning mode. '
    'A new one-minute piano performance with audio, every matched outcome and fixed-camera diagnostics appear below; no complete performance is accepted.')
# Keep one current research status when the publishing command is rerun.
manifest['status'] = [manifest['status'][0]] + [s for s in manifest['status'][1:] if not s.startswith('<b>Rig research:</b>')]
if composure is not None and composure['status'] == 'complete':
    manifest['updated'] = '2026-10-11 curled-finger / wrist-carry performance comparison; complete realism remains unfinished'
    manifest['status'][0] = (f"<b>Rig research:</b> {composure['unit_tests']['passed']} tests pass. "
        'Fresh grayscale controller comparison and original-score hold coverage are published. '
        'Free-finger curl and isolated wrist carry remain experimental; skin and musical failures are retained. '
        'The one-minute performance and drummer diagnostics remain available below.')
media = WEB / 'media'
media.mkdir(exist_ok=True)
if key_action is not None and key_action['status'] == 'complete':
    for source, destination in ((ART/'key_action_semantics.png','rig_realism_key_action_semantics.png'),
                                (key_action_path,'rig_realism_key_action_results.json'),
                                (ART/'key_action_recovered_pedal.json','rig_realism_recovered_pedal.json')):
        shutil.copy2(source, media/destination)
if orientation is not None and orientation['status'] == 'complete':
    for source, destination in ((ART/'orientation_comparison.mp4','rig_realism_orientation_comparison.mp4'),
                                (ART/'orientation_traces.png','rig_realism_orientation_traces.png'),
                                (orientation_path,'rig_realism_orientation_results.json')):
        shutil.copy2(source, media/destination)
if continuous_plan is not None and continuous_plan['status'] == 'complete':
    for source, destination in ((ART/'continuous_plan_comparison.mp4','rig_realism_continuous_plan_comparison.mp4'),
                                (continuous_plan_path,'rig_realism_continuous_plan_results.json')):
        shutil.copy2(source, media/destination)
if full_suite is not None and full_suite['status'] == 'complete':
    for source, destination in ((ART/'full_suite_progress_preview.mp4','rig_realism_full_suite_progress.mp4'),
                                (full_suite_path,'rig_realism_full_suite_preview_results.json')):
        shutil.copy2(source, media/destination)
if hand_order is not None and hand_order['status'] == 'complete':
    manifest['updated'] = '2026-10-11 ordered-hand one-minute performance; complete rig realism remains unfinished'
    manifest['status'][0] = (f"<b>Rig research:</b> {hand_order['unit_tests']['passed']} tests pass. "
        'The fresh minute keeps left/right wrist order across all 1,440 solved samples. '
        'Original music, sampled skin failures and chunk-boundary limits are reported below. '
        'A root-order pass does not accept full realistic motion.')
    for source, destination in (
            (ART / 'hand_order_minute.mp4', 'rig_realism_ordered_piano_60s.mp4'),
            (ART / 'hand_order_comparison.mp4', 'rig_realism_hand_order_comparison.mp4'),
            (ART / 'hand_order_metrics.png', 'rig_realism_hand_order_metrics.png'),
            (hand_order_path, 'rig_realism_hand_order_results.json')):
        shutil.copy2(source, media / destination)
if composure is not None and composure['status'] == 'complete':
    for source, destination in (
            (ART / 'composure_comparison.mp4', 'rig_realism_composure_comparison.mp4'),
            (ART / 'composure_metrics.png', 'rig_realism_composure_metrics.png'),
            (composure_path, 'rig_realism_composure_results.json'),
            (ROOT / 'lab/realism_results/composure_review_result.json', 'rig_realism_composure_review.json')):
        shutil.copy2(source, media / destination)
for source, destination in (
        (ART / 'performance_60s.mp4', 'rig_realism_piano_60s.mp4'),
        (ROOT / 'lab/realism_results/performance_preview_20261011.json', 'rig_realism_performance_preview_results.json'),
        (ART / 'deformation_and_union.png', 'rig_realism_deformation_and_union.png'),
        (ART / 'fold_verify_20261011/R_sample0.png', 'rig_realism_ring_uncurled.png'),
        (ART / 'fold_verify_20261011/R_sample5.png', 'rig_realism_ring_overcurled.png'),
        (ROOT / 'lab/realism_results/deformation_and_union_20261011.json', 'rig_realism_deformation_and_union_results.json'),
        (ROOT / 'lab/realism_results/deformation_review_result.json', 'rig_realism_deformation_review.json'),
        (ART / 'video_measured_piano.mp4', 'rig_realism_measured_piano_diagnostic.mp4'),
        (ART / 'drums/crosshand_demo_fixed.mp4', 'rig_realism_drummer_fixed.mp4'),
        (ART / 'mesh_waypoint_experiment.png', 'rig_realism_mesh_waypoint_experiment.png'),
        (ROOT / 'lab/realism_results/mesh_waypoint_20261011.json', 'rig_realism_mesh_waypoint_results.json'),
        (ROOT / 'lab/realism_results/form_mesh_review_result.json', 'rig_realism_form_mesh_review.json'),
        (ART / 'human_contact_pilot.png', 'rig_realism_human_contact_pilot.png'),
        (ART / 'human_pilot_source_fixed/f_00024.png', 'rig_realism_measured_contact_only.png'),
        (ART / 'human_pilot_form_fixed/f_00024.png', 'rig_realism_measured_form_guidance.png'),
        (ROOT / 'lab/realism_results/human_contact_pilot_20261011.json', 'rig_realism_human_contact_results.json'),
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
if continuous_plan is not None and continuous_plan['status'] == 'complete':
    manifest['status'][0] = (f"<b>Rig research:</b> {continuous_plan['unit_tests']['passed']} tests pass. "
        'A single full-score plan and chronological solve remove the sampled join discontinuity in the tested passage. '
        'All controller treatments, physical warmup and actual-skin failures are published; full realism remains unfinished.')
if full_suite is not None and full_suite['status'] == 'complete':
    manifest['updated'] = '2026-10-11 entire 6:34 Knights Suite preview with original audio and updated wrist-order minute; full realism remains unfinished'
if orientation is not None and orientation['status'] == 'complete':
    manifest['updated'] = '2026-10-11 full-suite progress preview and matched wrist-orientation experiment; full realism remains unfinished'
    manifest['status'][0] = (f"<b>Rig research:</b> {orientation['unit_tests']['passed']} tests pass. "
        'The full 6:34 preview and four matched wrist-orientation treatments are available below. '
        'Original music, sampled skin failures and physical warmup remain in the evidence; full realism is unfinished.')
if key_action is not None and key_action['status'] == 'complete':
    manifest['updated'] = '2026-10-11 full-suite preview, recovered pedal estimates and complete source-action audit; realism remains unfinished'
    manifest['status'][0] = (f"<b>Rig research:</b> {key_action['unit_tests']['passed']} tests pass. "
        'The 6:34 suite preview and controller comparisons remain available. '
        'Recovered pedal evidence exposes unverified hold cuts and two unscheduled source attacks. '
        'Physical key/foot actions, geometry and natural performance remain unfinished.')
manifest_path.write_text(json.dumps(manifest, indent=1) + '\n')
print('Updated existing band-site manifest and research media')
