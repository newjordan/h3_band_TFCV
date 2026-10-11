"""Record every static depth/self-mesh treatment and independent mesh audit.

This sensitivity pilot cannot certify a connected trajectory or natural form.
Run after rendering/encoding the diagnostic samples; source poses stay frozen.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART, OUT = ROOT / 'artifacts/realism', ROOT / 'lab/realism_results'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


names = ('hard_proxy', 'hard_mesh', 'soft_proxy', 'soft_mesh')
trials = {n: read(ART / f'waypoint_{n}.json') for n in names}
base = trials[names[0]]
for trial in trials.values():
    for key in ('source_sha256', 'code_sha256', 'score_sha256', 'hand_scale', 'geometry',
                'required_pitches', 'key_commands', 'budget', 'seed', 'wrist_workspace_m'):
        assert trial[key] == base[key], key
    assert trial['source_unchanged_during_run']
    assert not trial['accepted'] and not trial['animation_candidate_exported']
    assert not trial['temporal_bounds_enforced'] and not trial['performance_accepted']

summaries = {n: dict(accepted=t['accepted'], instrument_depth_mm=t['final_penetration_mm'],
    instrument_triangle_intersections=t['triangle_intersections'],
    same_hand_triangle_pair_intersections=t['mesh_self']['unexpected_surface_intersections'],
    proxy_clearance_mm=t['proxy_self_clearance_mm'],
    supported_anchors=sum(a['exposed_top_support'] for a in t['anchors']),
    requested_anchors=len(t['anchors']),
    soft_form_cost=t['human_form']['numerical_iterate']['cost'], seconds=t['seconds'])
    for n, t in trials.items()}

actual = {n: read(ART / f'skin_mesh_measured_{n}.json') for n in ('source', 'form')}
cpu = read(ART / 'skin_mesh_cpu_measured_form.json')
bvh_pairs = {tuple(sorted(p)) for p in actual['form']['records'][0]['unexpected_pairs']}
cpu_pairs = {tuple(sorted(p)) for p in cpu['pairs']}
assert bvh_pairs <= cpu_pairs
extra = sorted(cpu_pairs - bvh_pairs)
assert extra == [(105697, 105702)]
assert not any(a['surface_intersection_clear'] for a in actual.values())
fixture = ROOT / 'tests/fixtures/skin_fold_shared_vertex.json'

review = read(OUT / 'form_mesh_review_result.json')
episode = read(OUT / 'form_mesh_review_episode.json')
assert review['status'] == 'completed' and review['stop_reason'] == 'answer'
assert episode['termination']['kind'] == 'completed' and len(episode['traces']) == 5
assert all(t['termination']['kind'] == 'answer' for t in episode['traces'])
assert 'PASS for the scoped method' in review['answer']

fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), layout='constrained')
for label, suffix, color in [('Active proxy constraints', 'proxy', '#c86a3a'),
                              ('Active mesh + proxy constraints', 'mesh', '#2b849f')]:
    points = [summaries[f'{depth}_{suffix}'] for depth in ('hard', 'soft')]
    for ax, metric in zip(axes, ('instrument_depth_mm', 'same_hand_triangle_pair_intersections',
                                 'supported_anchors')):
        ax.plot([0, 1], [p[metric] for p in points], 'o-', color=color, label=label)
        ax.set(xticks=[0, 1], xticklabels=['Hard pooled median ±2 mm', 'Soft pooled depth band'])
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(alpha=.2)
axes[0].set(ylabel='Instrument penetration (mm)', yscale='log')
axes[0].axhline(.01, color='#777777', linestyle='--', label='Instrument tolerance (0.01 mm)')
axes[1].set(ylabel='Same-hand crossing triangle pairs (max of 2 diagonals)')
axes[2].set(ylabel='Independently supported anchors / 4', ylim=(-.15, 4.3))
axes[2].axhline(4, color='#777777', linestyle='--')
axes[0].legend(fontsize=7)
fig.suptitle('Static waypoint sensitivity: all four treatments rejected; no motion or pose export')
fig.savefig(ART / 'mesh_waypoint_experiment.png', dpi=160)
plt.close(fig)


def video(source, path, camera, status):
    data = read(source)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,avg_frame_rate,nb_frames,duration',
        '-show_entries', 'format=duration', '-of', 'json', str(path)], text=True))
    stream = probe['streams'][0]
    assert int(stream['nb_frames']) == len(data['frames'])
    assert stream['codec_name'] == 'h264'
    return dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source),
                video=str(path.relative_to(ROOT)), video_sha256=sha(path),
                frames=len(data['frames']), source_fps=data['fps'], probe=probe,
                camera=camera, status=status, audio=False,
                renderer_sha256=sha(ROOT / ('blockout/blender_piano.py' if 'piano' in path.name
                                            else 'blockout/drum_preview.py')))


videos = dict(piano=video(ART / 'measured_notched_overlap.anim.json',
    ART / 'video_measured_piano.mp4',
    dict(eye=[.35, -.52, .38], look=[0, .015, .02], lens_mm=42, fixed=True,
         resolution=[1280, 720], renderer='Blender 4.0.2 Workbench software', hides='body only'),
    'Rejected diagnostic baseline: 63/86 saved frames intersect instrument; no full-mesh motion pass'),
    drummer=video(ART / 'drums/crosshand_demo.json', ART / 'drums/crosshand_demo_fixed.mp4',
    dict(elevation_deg=27, azimuth_deg=-62, projection='orthographic', zoom=1.2, fixed=True,
         renderer='Matplotlib CPU; colored chains are proxies'),
    '10/10 demo impacts pass 240 Hz proxy audit; 60 fps preview; full skin/body/force/continuous-time unverified'))
report = dict(schema='mesh-waypoint-summary/v1', date='2026-10-11',
    hypothesis='Allowing depth contact to slide inside finite exposed key tops with soft human depth guidance can improve '
               'instrument clearance; actual mesh self constraints are needed beyond different-finger proxies.',
    ablation='Same measured-size R frame 24, source initialization, score/key commands, anatomy, 120 mm static wrist '
             'workspace, actual level-2 skin, human weight 4 and 100×4 plus 80 feasibility iterations per treatment. '
             'Hard/soft depth guidance × active self-pair constraints off/on. All four retain the identical final '
             'two-triangulation actual self-mesh rejection gate. No neighbor rate constraints in static planning.',
    scope='Exploratory static sensitivity pilot; not the graph-reviewed connected trajectory comparison',
    summaries=summaries, trials=trials, actual_rendered_mesh_audits=actual,
    earlier_hard_median_seed_searches={n: read(ART / f'waypoint_calibrated_{n}.json')
                                      for n in ('source', 'comfort')},
    cpu_actual_mesh_audit=cpu,
    bvh_candidate_counterexample=dict(extra_pairs=extra, fixture=str(fixture.relative_to(ROOT)),
        fixture_sha256=sha(fixture), intersection_curve_length_mm=.310599455792,
        meaning='Intersection segment length, not penetration depth; independent barycentric witness test passes.'),
    reviewed_method=dict(episode_id=episode['episode_id'], receipt_sha256=sha(OUT / 'form_mesh_review_episode.json'),
        five_terminal_nodes=True, verdict='PASS for connected experiment protocol only',
        connected_experiment_completed=False, local_review_result_sha256=sha(OUT / 'form_mesh_review_result.json')),
    unit_tests=dict(passed=130, seconds=39.110, log_sha256=sha(ART / 'tests_mesh_soft_contact.log')),
    videos=videos,
    supersedes='Earlier local accepted poses covered instrument, finite support, bounds and proxy spacing. '
               'They now fail actual same-hand skin clearance and cannot be called complete-hand passes.',
    conclusion='Soft depth guidance reduces instrument depth from roughly 3.04 mm to 0.0147 mm, but neither treatment '
               'clears instrument triangles, finger proxies, full self surfaces and all four anchors. '
               'Active self constraints do not resolve the folds within this budget. Every numerical trial is rejected.',
    full_performance_accepted=False,
    next_hypotheses=['Distinguish pose-induced folds from skin binding/deformation behavior with identical frozen poses.',
                     'Evaluate worst individual joint angles, not only aggregate reference deviation.',
                     'Plan connected contact/wrist paths with original motion limits and immutable musical ledger.',
                     'Replace drummer grip proxies with independently audited hand/body/shell geometry.'],
    exclusions=['positive minimum mesh distance', 'volume containment', 'full body/costume/drummer mesh',
                'complete musical onsets/accidentals/repertoire', 'forces/muscle/piano action',
                'natural motion acceptance', 'continuous collision detection'])
OUT.mkdir(exist_ok=True)
(OUT / 'mesh_checks_tests_20261011.log').write_bytes((ART / 'tests_mesh_soft_contact.log').read_bytes())
(OUT / 'mesh_waypoint_20261011.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(dict(summaries=summaries, videos={k:v['probe'] for k,v in videos.items()},
                     full_performance_accepted=False), indent=2))
