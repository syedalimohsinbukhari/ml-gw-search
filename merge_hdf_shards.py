#! /usr/bin/env python
"""Merge repackaged shards (re_package_hdf_with_noise.py) into one dataset file.

Shards are sorted by (split, shard number), so the merged arrays do not depend on the order of the inputs (SEEDING.md, step 6).
The file is written under a temporary name and renamed on success, and `complete` is the last attribute set.
Arrays are copied shard by shard into pre-sized datasets, so memory stays at one shard.
The merged layout has, per split group, the datasets of a repackaged shard concatenated in shard order
(h1, l1, params, h1_unit_waveform, l1_unit_waveform for injection rows; noise_h1, noise_l1 for noise-only rows).
"""

import hashlib
import json
import os
import socket
import subprocess
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone

import h5py
import numpy as np

SPLIT_ORDER = ('training', 'validation', 'test')
SCHEMA_VERSION = 1
INJECTION_DATASETS = ('h1', 'l1', 'h1_unit_waveform', 'l1_unit_waveform', 'params')
NOISE_DATASETS = ('noise_h1', 'noise_l1')

# Attributes that must be identical in every shard: they define the generating distribution.
CONSTANT_ATTRS = ('schema_version', 'sample_length', 'sample_rate', 'detectors', 'approximant', 'noise_source',
                  'low_frequency_cutoff', 'segment_duration', 'max_filter_length', 'whitened', 'fft_backend',
                  'master_seed', 'dataset_id', 'columns', 'params_columns', 'backend_hashes', 'snr_range',
                  'convention')
# Library versions: also required to match unless --allow-version-mismatch, since they can change the numerics.
VERSION_ATTRS = ('python', 'numpy', 'pycbc', 'h5py', 'torch')
# Recorded per shard in the merged file's `shards` attribute.
PER_SHARD_ATTRS = ('git_sha', 'git_dirty', 'generated_utc', 'host', 'repackage_git_sha', 'repackage_git_dirty',
                   'repackage_utc', 'repackage_host', 'source_file', 'source_sha256', 'seed_keys', 'snr_seed_key')


def plain(value):
    """An h5py attribute as a JSON-friendly Python value."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read_header(path):
    with h5py.File(path, 'r') as f:
        if f.attrs.get('kind') != 'repackaged_shard':
            raise RuntimeError("%s is not a repackaged shard (kind=%r)" % (path, f.attrs.get('kind')))
        if not f.attrs.get('complete', False):
            raise RuntimeError("%s is not marked complete" % path)
        attrs = {k: plain(v) for k, v in f.attrs.items()}
        split = str(attrs['split'])
        if split not in SPLIT_ORDER or list(f.keys()) != [split]:
            raise RuntimeError("%s: unexpected split or groups %s" % (path, list(f.keys())))
        group = f[split]
        shapes = {k: group[k].shape for k in INJECTION_DATASETS + NOISE_DATASETS}
        dtypes = {k: group[k].dtype for k in INJECTION_DATASETS + NOISE_DATASETS}
    n_inj, n_noise = attrs['n_injections'], attrs['n_noise_only']
    length = attrs['sample_length']
    for name in INJECTION_DATASETS:
        if shapes[name][0] != n_inj:
            raise RuntimeError("%s: %s has %d rows, expected %d" % (path, name, shapes[name][0], n_inj))
    for name in NOISE_DATASETS:
        if shapes[name] != (n_noise, length):
            raise RuntimeError("%s: %s has shape %s, expected %s" % (path, name, shapes[name], (n_noise, length)))
    return {'path': path, 'split': split, 'shard': int(attrs['shard']), 'attrs': attrs, 'shapes': shapes,
            'dtypes': dtypes, 'n_inj': int(n_inj), 'n_noise': int(n_noise)}


def check_shards(headers, allow_version_mismatch, expect):
    by_split = {}
    for h in headers:
        by_split.setdefault(h['split'], []).append(h)
    for split, group in by_split.items():
        numbers = sorted(h['shard'] for h in group)
        if numbers != list(range(len(numbers))):
            raise RuntimeError("%s shards are %s, expected 0 to %d with no gaps or duplicates" %
                               (split, numbers, len(numbers) - 1))
    if expect:
        found = {s: len(g) for s, g in by_split.items()}
        if found != expect:
            raise RuntimeError("Shard counts per split are %s, expected %s" % (found, expect))
    first = headers[0]
    names = CONSTANT_ATTRS + (() if allow_version_mismatch else VERSION_ATTRS)
    for h in headers[1:]:
        for name in names:
            if h['attrs'].get(name) != first['attrs'].get(name):
                raise RuntimeError("Attribute %r differs: %s has %r, %s has %r" %
                                   (name, first['path'], first['attrs'].get(name), h['path'], h['attrs'].get(name)))
        if h['shapes']['h1'][1:] != first['shapes']['h1'][1:] or h['shapes']['params'][1:] != first['shapes']['params'][1:]:
            raise RuntimeError("%s: row shapes differ from %s" % (h['path'], first['path']))
        if h['dtypes'] != first['dtypes']:
            raise RuntimeError("%s: dtypes differ from %s" % (h['path'], first['path']))
    return by_split


def git_sha():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(('git', '-C', here, 'rev-parse', 'HEAD'), capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(('git', '-C', here, 'status', '--porcelain', '--untracked-files=no'),
                                    capture_output=True, text=True, check=True).stdout.strip())
        return sha, dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown', True


def merge(headers, output, allow_version_mismatch=False, expect=None, force=False, argv=None):
    by_split = check_shards(headers, allow_version_mismatch, expect)
    if os.path.exists(output) and not force:
        raise RuntimeError("Output exists: %s" % output)
    partial = output + '.partial'
    first = headers[0]
    shard_records = []
    sha, dirty = git_sha()

    with h5py.File(partial, 'w') as out:
        for name in CONSTANT_ATTRS + VERSION_ATTRS:
            if name in first['attrs']:
                out.attrs[name] = first['attrs'][name]
        out.attrs['kind'] = 'merged'
        out.attrs['splits'] = json.dumps([s for s in SPLIT_ORDER if s in by_split])
        out.attrs['merge_order'] = 'by split (training, validation, test), then shard number'
        for split in SPLIT_ORDER:
            if split not in by_split:
                continue
            group_headers = sorted(by_split[split], key=lambda h: h['shard'])
            n_inj = sum(h['n_inj'] for h in group_headers)
            n_noise = sum(h['n_noise'] for h in group_headers)
            group = out.create_group(split)
            group.attrs['n_injections'] = n_inj
            group.attrs['n_noise_only'] = n_noise
            group.attrs['shards'] = json.dumps([h['shard'] for h in group_headers])
            for name in INJECTION_DATASETS:
                shape = (n_inj,) + first['shapes'][name][1:]
                group.create_dataset(name, shape=shape, dtype=first['dtypes'][name])
            for name in NOISE_DATASETS:
                group.create_dataset(name, shape=(n_noise,) + first['shapes'][name][1:], dtype=first['dtypes'][name])
            inj_at = noise_at = 0
            for h in group_headers:
                with h5py.File(h['path'], 'r') as src:
                    for name in INJECTION_DATASETS:
                        group[name][inj_at:inj_at + h['n_inj']] = src[split][name][()]
                    for name in NOISE_DATASETS:
                        group[name][noise_at:noise_at + h['n_noise']] = src[split][name][()]
                    if inj_at == 0 and 'columns' in src[split]['params'].attrs:
                        group['params'].attrs['columns'] = list(src[split]['params'].attrs['columns'])
                record = {'split': split, 'shard': h['shard'], 'file': os.path.basename(h['path']),
                          'file_sha256': sha256_of(h['path']), 'n_injections': h['n_inj'], 'n_noise_only': h['n_noise'],
                          'injection_rows': [inj_at, inj_at + h['n_inj']],
                          'noise_rows': [noise_at, noise_at + h['n_noise']]}
                record.update({k: h['attrs'][k] for k in PER_SHARD_ATTRS if k in h['attrs']})
                shard_records.append(record)
                inj_at += h['n_inj']
                noise_at += h['n_noise']
        out.attrs['shards'] = json.dumps(shard_records)
        out.attrs['merge_git_sha'] = sha
        out.attrs['merge_git_dirty'] = dirty
        out.attrs['merge_utc'] = datetime.now(timezone.utc).isoformat()
        out.attrs['merge_host'] = socket.gethostname()
        out.attrs['merge_argv'] = json.dumps(argv if argv is not None else sys.argv)
        out.attrs['complete'] = True
    os.replace(partial, output)


def main():
    parser = ArgumentParser(description="Merge repackaged shards into one dataset file.")
    parser.add_argument('--inputs', nargs='+', required=True, help="Repackaged shard files, in any order.")
    parser.add_argument('--output', required=True)
    parser.add_argument('--expect', nargs='+', default=None, metavar='SPLIT=COUNT',
                        help="Required shards per split, for example training=10 validation=2 test=1.")
    parser.add_argument('--allow-version-mismatch', action='store_true',
                        help="Accept shards generated with different library versions.")
    parser.add_argument('--force', action='store_true', help="Overwrite an existing output file.")
    args = parser.parse_args()

    expect = {k: int(v) for k, v in (item.split('=') for item in args.expect)} if args.expect else None
    headers = [read_header(p) for p in args.inputs]
    headers.sort(key=lambda h: (SPLIT_ORDER.index(h['split']), h['shard']))
    merge(headers, args.output, args.allow_version_mismatch, expect, args.force)
    print("%s: %s" % (args.output, ', '.join('%s %d shards' % (s, sum(h['split'] == s for h in headers))
                                             for s in SPLIT_ORDER if any(h['split'] == s for h in headers))))


if __name__ == '__main__':
    main()
