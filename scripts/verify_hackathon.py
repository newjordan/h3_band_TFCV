#!/usr/bin/env python3
"""Bounded stdlib-only checks. Never imports project/runtime dependencies."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / 'examples/workflows/hackathon'
OUTPUTS = {
    'UNETLoader': 1, 'CLIPLoader': 1, 'VAELoader': 1, 'ModelPatchLoader': 1,
    'LoadVideo': 1, 'GetVideoComponents': 3, 'LoadAudio': 1, 'ImageFromBatch': 1,
    'VAEEncode': 1, 'MiniMaxH3FunControlNetApply': 1,
    'MiniMaxH3ImageToVideo': 2, 'MiniMaxH3ReferenceToVideo': 2,
    'H3BandZoneLatent': 1, 'KSamplerSelect': 1, 'BasicScheduler': 1,
    'RandomNoise': 1, 'BasicGuider': 1, 'SamplerCustomAdvanced': 2,
    'VAEDecode': 1, 'VAEDecodeAudio': 1, 'CreateVideo': 1, 'SaveVideo': 0,
}
REQUIRED = {
    'UNETLoader': ['unet_name', 'weight_dtype'],
    'CLIPLoader': ['clip_name', 'type', 'device'], 'VAELoader': ['vae_name'],
    'ModelPatchLoader': ['name'], 'LoadVideo': ['file'], 'LoadAudio': ['audio'],
    'GetVideoComponents': ['video'], 'ImageFromBatch': ['image', 'batch_index', 'length'],
    'VAEEncode': ['pixels', 'vae'],
    'MiniMaxH3FunControlNetApply': ['model', 'model_patch', 'vae', 'strength', 'start_percent', 'end_percent', 'control_video'],
    'MiniMaxH3ImageToVideo': ['clip', 'vae', 'prompt', 'width', 'height', 'length'],
    'MiniMaxH3ReferenceToVideo': ['clip', 'vae', 'audio_vae', 'prompt', 'width', 'height', 'length', 'ref_image_size', 'ref_videos.ref_video_0'],
    'H3BandZoneLatent': ['latent', 'audio_vae', 'audio', 'video_denoise', 'audio_denoise'],
    'KSamplerSelect': ['sampler_name'], 'BasicScheduler': ['model', 'scheduler', 'steps', 'denoise'],
    'RandomNoise': ['noise_seed'], 'BasicGuider': ['model', 'conditioning'],
    'SamplerCustomAdvanced': ['noise', 'guider', 'sampler', 'sigmas', 'latent_image'],
    'VAEDecode': ['samples', 'vae'], 'VAEDecodeAudio': ['samples', 'vae'],
    'CreateVideo': ['images', 'audio', 'fps'], 'SaveVideo': ['video', 'filename_prefix', 'format', 'codec'],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    require(path.stat().st_size < 2_000_000, 'Oversize handoff file: ' + str(path))
    return json.loads(path.read_text())


def validate_api(graph):
    require(isinstance(graph, dict) and graph, 'Empty/non-dictionary graph')
    dependencies = {}
    for key, node in graph.items():
        require(isinstance(key, str) and key.isdigit(), 'Non-numeric node ID')
        cls = node['class_type']
        require(cls in OUTPUTS, 'Unknown node class: ' + cls)
        inputs = node['inputs']
        require(set(REQUIRED[cls]) <= set(inputs), 'Missing required inputs for ' + cls)
        dependencies[key] = []
        for name, value in inputs.items():
            if isinstance(value, list):
                require(len(value) == 2, 'Invalid edge: ' + name)
                src, slot = value
                require(src in graph, 'Dangling edge: ' + str(src))
                require(type(slot) is int and 0 <= slot < OUTPUTS[graph[src]['class_type']], 'Invalid output slot')
                dependencies[key].append(src)
            elif name in ('file', 'audio', 'image', 'filename_prefix'):
                require(isinstance(value, str) and not value.startswith(('/', '~')) and '..' not in Path(value).parts, 'Nonportable path')
    visiting, visited = set(), set()
    def walk(key):
        require(key not in visiting, 'Graph cycle')
        if key in visited:
            return
        visiting.add(key)
        for src in dependencies[key]:
            walk(src)
        visiting.remove(key)
        visited.add(key)
    for key in graph:
        walk(key)
    require(any(n['class_type'] == 'SaveVideo' for n in graph.values()), 'No video output')


def validate_canvas(canvas, api):
    require(canvas['version'] == 0.4, 'Unexpected canvas version')
    nodes = {n['id']: n for n in canvas['nodes']}
    require(set(nodes) == {int(k) for k in api}, 'Canvas/API IDs differ')
    for key, node in api.items():
        require(nodes[int(key)]['type'] == node['class_type'], 'Canvas/API class differs')
        expected = [v for v in node['inputs'].values() if not isinstance(v, list)]
        require(nodes[int(key)]['widgets_values'] == expected, 'Canvas scalar mismatch')
    seen = set()
    for lid, src, slot, dst, inp, typ in canvas['links']:
        require(lid not in seen, 'Duplicate canvas link')
        seen.add(lid)
        require(nodes[src]['outputs'][slot]['type'] == typ, 'Canvas type mismatch')
        socket = nodes[dst]['inputs'][inp]
        require(socket['link'] == lid and socket['type'] == typ, 'Canvas input mismatch')
        require(lid in nodes[src]['outputs'][slot]['links'], 'Canvas output mismatch')
        name = socket['name']
        require(api[str(dst)]['inputs'][name] == [str(src), slot], 'Canvas/API edge mismatch')
    expected_count = sum(isinstance(v, list) for n in api.values() for v in n['inputs'].values())
    require(len(seen) == expected_count, 'Missing canvas links')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, help='Optional small JSON receipt path')
    args = parser.parse_args()
    checks = []
    for name in ['fl2va_control', 'ref2va_video_reference']:
        api = load(WORKFLOWS / (name + '_api.json'))
        validate_api(api)
        validate_canvas(load(WORKFLOWS / (name + '_canvas.json')), api)
        if name.startswith('ref2va'):
            require(not any(n['class_type'] == 'MiniMaxH3FunControlNetApply' for n in api.values()), 'Unqualified Ref2VA ControlNet')
            require('ref_videos.ref_video_0' in api['7']['inputs'], 'Missing actual video reference')
            require('ref2va' in api['1']['inputs']['unet_name'], 'Wrong Ref2VA checkpoint')
            require(api['16']['inputs']['audio'] == ['6', 0], 'Ref2VA soundtrack contract changed')
        checks.append(name + ': API topology, required inputs, relative paths and canvas parity PASS')
    baseline = load(WORKFLOWS / 'fl2va_control_api.json')
    for label, change in [
        ('dangling edge', lambda g: g['16']['inputs'].update(images=['999', 0])),
        ('bad slot', lambda g: g['16']['inputs'].update(images=['14', 9])),
        ('cycle', lambda g: g['14']['inputs'].update(samples=['14', 0])),
        ('missing socket', lambda g: g['16']['inputs'].pop('images')),
    ]:
        broken = copy.deepcopy(baseline)
        change(broken)
        try:
            validate_api(broken)
        except ValueError:
            checks.append('Negative control rejected: ' + label)
        else:
            raise ValueError('Verifier accepted ' + label)
    plan = load(ROOT / 'hackathon/reference_match/test_plan.json')
    require(plan['execution_enabled'] is False and plan['status'] == 'PREPARED_NOT_LAUNCHED', 'Launch must remain disabled')
    require(plan['dense_frame_count'] == plan['original_frame_count'] * plan['density'], 'Frame clock mismatch')
    checks.append('Reference plan launch-disabled and dense/original count invariants PASS')
    expected = {'H3BandZoneLatent', 'BandAidBlockoutPoseControl', 'BandAidBlockoutReferenceEdges', 'BandAidBlockoutLockCheck', 'H3BandDrumEvents', 'H3BandDrumHitsMIDI', 'H3BandDrumBlockout'}
    found = set()
    for path in (ROOT / 'comfy_nodes').glob('*.py'):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'NODE_CLASS_MAPPINGS' for t in node.targets):
                if isinstance(node.value, ast.Dict):
                    found.update(k.value for k in node.value.keys if isinstance(k, ast.Constant))
    require(expected <= found, 'Missing advertised custom-node mappings')
    for name in ['verify_hackathon.py', 'package_hackathon.py']:
        ast.parse((ROOT / 'scripts' / name).read_text())
    checks.append('Seven custom-node mappings and utility syntax PASS (AST only; no runtime imports)')
    files = list((ROOT / 'hackathon').rglob('*.md')) + [WORKFLOWS / 'README.md']
    for path in files:
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            if not link.startswith(('https://', 'http://', '#')):
                require((path.parent / link.split('#')[0]).exists(), 'Broken local documentation link: ' + link)
    require((ROOT / 'docs/hackathon.html').is_file(), 'Missing public landing page')
    require((ROOT / 'LICENSE').is_file() and (ROOT / '__init__.py').is_file(), 'Missing package entry/license')
    checks.append('Local Markdown links, landing page, plugin entry and license PASS')
    paths = sorted(set(list((ROOT / 'hackathon').rglob('*')) + list(WORKFLOWS.glob('*')) + list((ROOT / 'scripts').glob('*')) + [ROOT / 'docs/hackathon.html']))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file() and p != args.receipt and p.name != 'verification.json'}
    receipt = {'result': 'PASS_OFFLINE_HANDOFF', 'checks': checks, 'sha256': hashes,
               'not_verified': ['Target Comfy schema/UI import', 'Model inference or fidelity', 'Fresh install or Registry publication', 'Clip rights/provenance'],
               'resource_scope': 'stdlib-only small file/AST/JSON checks; no model, tensor, media decode, GPU, services or network'}
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, SyntaxError) as error:
        print('FAIL_OFFLINE_HANDOFF: ' + str(error), file=sys.stderr)
        sys.exit(1)
