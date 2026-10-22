#! /usr/bin/env python
"""Repackage one raw shard from mlgwsc-1/gen_seeded.py into the consumer layout, with a baked SNR and a noise-only half.

The original re_package_hdf.py stays unchanged; this is its seeded successor (../ml_practice_v2/GENERATOR_OVERHAUL.md, D-04, D-05, D-12).
One shard in, one shard out, so the SNRs of a shard depend only on its own key (SEEDING.md, purpose 2).

Output group <split>, for n_inj injection rows and n_noise noise-only rows:

    h1, l1                       (n_inj, L)   float32  strain = noise + snr * unit-SNR waveform, whitened
    h1_unit_waveform,
    l1_unit_waveform             (n_inj, L)   float32  whitened waveform at unit network SNR
    params                       (n_inj, 10)  float64  the nine generator columns, then snr
    noise_h1, noise_l1           (n_noise, L) float32  noise-only strain, whitened

The loader in ml_practice_v2 reads only h1, l1, and params, so it works unchanged.
The scaled waveform is snr * unit waveform and the pure noise of an injection row is h1 - snr * h1_unit_waveform;
the latter carries float32 rounding error, which is small next to the noise.
"""

import hashlib
import json
import os
import socket
import subprocess
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlgwsc-1'))

import h5py
import numpy as np

import gen_seeded

SNR_RANGE = (7., 15.)
CONVENTION = 'strain = noise + snr * unit_waveform, both whitened; snr is the network SNR'
REQUIRED_ATTRS = ('complete', 'master_seed', 'dataset_id', 'split', 'shard', 'n_injections', 'n_noise_only', 'columns')


def draw_snr(dataset_id, split, shard, n, snr_range=SNR_RANGE):
    rng = np.random.default_rng(gen_seeded.seed_seq(dataset_id, split, shard, gen_seeded.PURPOSE_SNR))
    return rng.uniform(*snr_range, size=n)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def repackage(noises, waveforms, waveform_params, snr):
    """Return the output arrays; noises holds the injection rows first, then the noise-only rows."""
    n_inj = len(waveforms)
    inj_noise = noises[:n_inj].astype(np.float64)
    strain = inj_noise + snr[:, None, None] * waveforms.astype(np.float64)
    params = np.column_stack([waveform_params.astype(np.float64), snr])
    return {
        'h1': strain[:, 0].astype(np.float32),
        'l1': strain[:, 1].astype(np.float32),
        'h1_unit_waveform': waveforms[:, 0].astype(np.float32),
        'l1_unit_waveform': waveforms[:, 1].astype(np.float32),
        'params': params,
        'noise_h1': noises[n_inj:, 0].astype(np.float32),
        'noise_l1': noises[n_inj:, 1].astype(np.float32),
    }


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


def main():
    parser = ArgumentParser(description="Repackage one raw shard into the consumer layout with baked SNR and noise-only rows.")
    parser.add_argument('--input', required=True, help="Raw shard written by gen_seeded.py.")
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--snr-range', type=float, nargs=2, default=list(SNR_RANGE), metavar=('LOW', 'HIGH'))
    parser.add_argument('--force', action='store_true', help="Overwrite an existing output file.")
    args = parser.parse_args()
    snr_range = tuple(args.snr_range)

    with h5py.File(args.input, 'r') as src:
        missing = [a for a in REQUIRED_ATTRS if a not in src.attrs]
        if missing:
            raise RuntimeError("Input lacks attributes %s" % missing)
        if not src.attrs['complete']:
            raise RuntimeError("Input is not marked complete: %s" % args.input)
        if src.attrs['master_seed'] != gen_seeded.MASTER_SEED:
            raise RuntimeError("Input master_seed %s differs from this repository's %s" %
                               (src.attrs['master_seed'], gen_seeded.MASTER_SEED))
        dataset_id, shard = int(src.attrs['dataset_id']), int(src.attrs['shard'])
        split = str(src.attrs['split'])
        n_inj, n_noise = int(src.attrs['n_injections']), int(src.attrs['n_noise_only'])
        columns = json.loads(src.attrs['columns'])
        noises, waveforms = src[split]['noises'][()], src[split]['waveforms'][()]
        waveform_params = src[split]['waveform_params'][()]
        attrs = dict(src.attrs)

    if (len(waveforms), len(noises)) != (n_inj, n_inj + n_noise) or waveform_params.shape != (n_inj, len(columns)):
        raise RuntimeError("Array shapes %s, %s, %s disagree with the attributes (%d + %d, %d columns)" %
                           (waveforms.shape, noises.shape, waveform_params.shape, n_inj, n_noise, len(columns)))

    snr = draw_snr(dataset_id, split, shard, n_inj, snr_range)
    arrays = repackage(noises, waveforms, waveform_params, snr)

    final = os.path.join(args.out_dir, 'd%d_%s_%03d_repackaged.hdf' % (dataset_id, split, shard))
    partial = final + '.partial'
    if os.path.exists(final) and not args.force:
        raise RuntimeError("Output exists: %s" % final)
    os.makedirs(args.out_dir, exist_ok=True)

    sha, dirty = git_sha()
    with h5py.File(partial, 'w') as out:
        for name, value in attrs.items():
            if name not in ('kind', 'complete'):
                out.attrs[name] = value
        out.attrs['kind'] = 'repackaged_shard'
        out.attrs['source_file'] = os.path.basename(args.input)
        out.attrs['source_sha256'] = sha256_of(args.input)
        out.attrs['snr_range'] = np.array(snr_range)
        out.attrs['snr_seed_key'] = json.dumps(list(gen_seeded.seed_seq(
            dataset_id, split, shard, gen_seeded.PURPOSE_SNR).spawn_key))
        out.attrs['convention'] = CONVENTION
        out.attrs['params_columns'] = json.dumps(columns + ['snr'])
        out.attrs['repackage_git_sha'] = sha
        out.attrs['repackage_git_dirty'] = dirty
        out.attrs['repackage_utc'] = datetime.now(timezone.utc).isoformat()
        out.attrs['repackage_host'] = socket.gethostname()
        out.attrs['repackage_argv'] = json.dumps(sys.argv)
        group = out.create_group(split)
        for name, value in arrays.items():
            group.create_dataset(name, data=value)
        group['params'].attrs['columns'] = columns + ['snr']
        out.attrs['complete'] = True
    os.replace(partial, final)
    print("%s: %d injection rows, %d noise-only rows" % (final, n_inj, n_noise))


if __name__ == '__main__':
    main()
