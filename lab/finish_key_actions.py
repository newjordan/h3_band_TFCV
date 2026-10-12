"""Seal source recovery, unchanged target control, tests and methodological receipt."""
import hashlib
from collections import Counter
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/realism'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    path=ROOT/'lab/realism_results/key_actions_20261011.json'
    report=json.loads(path.read_text())
    assert report['code_unchanged_during_run']
    assert all(sha(ROOT/p)==h for p,h in report['input_sha256'].items())
    summary=report['source_actions']
    assert summary['source_attacks']==2046 and summary['all_attacks_accounted']
    assert summary['scheduled_actions']==2044 and summary['missing_source_ids']==[828,943]
    assert not summary['scheduling_bijection']
    recovery=report['source_recovery']
    assert recovery['all_raw_cc_accounted'] and recovery['raw_cc64_count']==282
    assert recovery['chopin_bytes_identical'] and recovery['suite_bytes_identical']
    control_path=ART/'key_action_target_control.json';control=json.loads(control_path.read_text())
    assert control['status']=='complete' and len(control['records'])==3
    assert all(r['all_preexisting_fields_equal'] for r in control['records'])
    log=ART/'key_action_tests.log';text=log.read_text()
    count=re.search(r'Ran (\d+) tests',text)
    assert count and int(count[1])==176 and text.rstrip().endswith('OK')
    episode_path=ROOT/'lab/realism_results/key_actions_review_episode.json'
    result_path=ROOT/'lab/realism_results/key_actions_review_result.json'
    episode=json.loads(episode_path.read_text());result=json.loads(result_path.read_text())
    assert episode['termination']['phase']=='done' and result['status']=='completed'
    assert {r['role'] for r in episode['traces']}=={'synthesize','verify'}
    plot_path=ART/'key_action_plot.json';plot=json.loads(plot_path.read_text())
    assert plot['status']=='complete' and plot['source_unchanged_during_run']
    assert sha(ART/'key_action_semantics.png')==plot['png_sha256']
    visual_path=ART/'key_action_visual_review.json';visual=json.loads(visual_path.read_text())
    assert visual['reviewed'] and visual['source_sha256']==plot['png_sha256']
    renderer={r['source_id']:r for r in report['sampler_damping_comparison']['records']}
    comparison=Counter()
    for record in report['estimated_pedal_release_audit']['records']:
        pedal='unknown' if record['covered'] is None else ('covered' if record['covered'] else 'not covered')
        audio='same sampler damping time' if renderer[record['source_id']]['same_damping_time'] else 'changed sampler damping time'
        comparison[(pedal,audio)]+=1
    assert sum(comparison.values())==len(summary['shortened_source_ids'])
    report.update(status='complete',native_graph_pending=False,
        unit_tests=dict(passed=int(count[1]),log=str(log),sha256=sha(log)),
        prepared_target_control=dict(control,source=str(control_path),sha256=sha(control_path)),
        plots=dict(plot,source=str(plot_path),sha256=sha(plot_path)),
        visual_review=dict(visual,source=str(visual_path),sha256=sha(visual_path)),
        joint_sustain_comparison=[dict(estimated_pedal=k[0],legacy_audio=k[1],proposals=v) for k,v in sorted(comparison.items())],
        unresolved=['Two original source attacks still lost at slicing; accounting does not reschedule them',
            'Predicted pedals are not ground truth and pedal-extended offsets do not uniquely identify key-off',
            'No verified checkpoint training invocation, per-event model confidence or half-pedal mechanics',
            'Earlier-piece and terminal-tail physical pedal states remain unverified',
            'Legacy Note interface lacks channel identity; recovered CC64 keeps channels separately',
            'No actual key/foot trajectory, acoustic-equivalence, force, skin/body/costume or nonlinear CCD acceptance'],
        native_research=dict(episode_id=episode['episode_id'],episode=str(episode_path),episode_sha256=sha(episode_path),
            result=str(result_path),result_sha256=sha(result_path),completed_nodes=2,
            scope='Completion metadata only; node text/output handle not disclosed. Final CLI proposal independently reviews the supplied brief, not an inspected reviewer consensus or rig acceptance'),
        conclusion='Recovered estimated pedal evidence and a complete source-action ledger expose unsupported releases and two omitted attacks. '
            'The original score, controller targets and soundtrack stay unchanged. Actual key/foot action and all rig realism remain unaccepted.')
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],unit_tests=int(count[1]),episode=episode['episode_id'],
        physical_key_actions_validated=False,full_performance_accepted=False)),flush=True)


if __name__=='__main__':main()
