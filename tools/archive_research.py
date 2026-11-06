"""Create a verified, immutable archive of existing research artifacts (no training)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUNS = ["risk_20261031T132513013959Z", "risk_20261031T162709361455Z", "risk_20261031T191741845697Z"]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def archive_group(files, base, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    inventory = destination.with_suffix('.inventory.json')
    if destination.exists() and inventory.exists():
        records = json.loads(inventory.read_text(encoding='utf-8'))
        actual = {f.relative_to(base).as_posix(): f.stat().st_size for f in files}
        if actual != {r['path']: r['bytes'] for r in records}:
            raise RuntimeError(f'Source inventory differs: {destination}')
        with zipfile.ZipFile(destination) as archive:
            for record in records:
                h = hashlib.sha256()
                with archive.open(record['path']) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(block)
                if h.hexdigest() != record['sha256']:
                    raise RuntimeError(f'Existing archive verification failed: {destination}')
        return {'path': destination.relative_to(ROOT).as_posix(), 'sha256': digest(destination),
                'bytes': destination.stat().st_size, 'files': len(records),
                'inventory': inventory.relative_to(ROOT).as_posix()}
    if destination.exists():
        if not destination.resolve().is_relative_to(ROOT / 'research_archive'):
            raise ValueError('Unsafe archive replacement path')
        destination.unlink()
    records = []
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for file in sorted(files):
            before = file.stat()
            h = hashlib.sha256()
            name = file.relative_to(base).as_posix()
            with file.open('rb') as source, archive.open(name, 'w', force_zip64=True) as target:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    h.update(block)
                    target.write(block)
            after = file.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Source changed during snapshot: {file}')
            records.append({'path': name, 'bytes': after.st_size, 'sha256': h.hexdigest()})
    # Independent decompression and per-file SHA256 comparison, not just ZIP CRC.
    with zipfile.ZipFile(destination) as archive:
        for record in records:
            h = hashlib.sha256()
            with archive.open(record['path']) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != record['sha256']:
                raise RuntimeError(f'Archive verification failed: {record["path"]}')
    inventory = destination.with_suffix('.inventory.json')
    inventory.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Verified {destination.name}: {len(records)} files, {destination.stat().st_size / 1e6:.1f} MB', flush=True)
    return {'path': destination.relative_to(ROOT).as_posix(), 'sha256': digest(destination),
            'bytes': destination.stat().st_size, 'files': len(records),
            'inventory': inventory.relative_to(ROOT).as_posix()}


def files_below(path):
    return [Path(base) / name for base, _, names in os.walk(path) for name in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', default='research_archive/snapshot_20261101')
    parser.add_argument('--resume', action='store_true', help='Continue an interrupted archive build, verifying existing ZIPs')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    output = (ROOT / args.destination).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Destination must be within repository')
    if args.resume and (output / 'manifest.json').exists():
        raise ValueError('Completed snapshots are immutable; use a new destination')
    output.mkdir(parents=True, exist_ok=args.resume)
    tasks = []
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'All files of three scientific runs, including search checkpoints; raw OULAD CSV inputs. No training or evaluation performed.',
                'archives': [], 'runs': []}
    for run in RUNS:
        source = ROOT / 'runs' / run
        metadata = json.loads((source / 'run_manifest.json').read_text(encoding='utf-8'))
        counts = {}
        for file in source.glob('day_*/fold_*/seed_*/*/status.json'):
            status = json.loads(file.read_text(encoding='utf-8')).get('status', 'unknown')
            counts[status] = counts.get(status, 0) + 1
        manifest['runs'].append({'run': run, 'status': metadata['status'], 'jobs': counts,
                                 'config': metadata['config'], 'signature': metadata['signature']})
        groups = {}
        for file in files_below(source):
            relative = file.relative_to(source)
            group = relative.parts[0] if relative.parts[0].startswith('day_') else 'metadata'
            groups.setdefault(group, []).append(file)
            # Human-readable reports are also available without downloading LFS.
            if len(relative.parts) == 1 or (len(relative.parts) == 2 and file.suffix in {'.csv', '.json'} and not file.name.endswith('_predictions.csv')):
                target = output / 'reports' / run / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
        for group, files in sorted(groups.items()):
            tasks.append((files, source, output / 'artifacts' / run / f'{group}.zip'))
    raw = ROOT / 'data' / 'oulad' / 'raw'
    tasks.append((files_below(raw), raw, output / 'artifacts' / 'oulad_raw.zip'))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        manifest['archives'] = list(executor.map(lambda task: archive_group(*task), tasks))
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('SNAPSHOT COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
