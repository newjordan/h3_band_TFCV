#!/usr/bin/env python3
"""Stream a committed source-only handoff archive; no installs/build/model imports."""
import argparse
import hashlib
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def selected(name):
    path = Path(name)
    if path.parts[0] in ('blockout', 'comfy_nodes', 'h3band', 'h3_tools'):
        return path.suffix in ('.py', '.json', '.md', '.txt', '.sh')
    if name.startswith(('hackathon/', 'examples/workflows/hackathon/', 'scripts/')):
        return path.suffix in ('.py', '.json', '.md', '.txt')
    return name in ('__init__.py', 'LICENSE', 'NOTICE_FurElise.md', 'pyproject.toml',
                    'requirements.txt', 'requirements-blockout.txt', 'requirements-sdk.txt',
                    'requirements-score.txt', 'requirements-sep.txt', 'docs/hackathon.html', 'README.md')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.with_name(output.name + '.sha256').exists():
        raise SystemExit('Refusing to overwrite an existing archive/checksum')
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'HEAD'], cwd=ROOT, text=True).splitlines()
    names = [n for n in names if selected(n)]
    if 'hackathon/README.md' not in names or 'scripts/verify_hackathon.py' not in names:
        raise SystemExit('Commit the handoff before packaging; archive uses HEAD, not dirty files')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        subprocess.run(['git', 'archive', '--format=tar.gz', '--prefix=band-aid/', 'HEAD', '--', *names], cwd=ROOT, stdout=stream, check=True)
    digest = hashlib.sha256()
    with output.open('rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            digest.update(chunk)
    output.with_name(output.name + '.sha256').write_text(digest.hexdigest() + '  ' + output.name + '\n')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    print('PACKAGED_COMMITTED_SOURCE', commit, len(names), 'files', output.stat().st_size, 'bytes')
    print(digest.hexdigest(), output)


if __name__ == '__main__':
    main()
