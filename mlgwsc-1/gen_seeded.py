#! /usr/bin/env python
"""Seeded driver for one shard of the regenerated dataset.

Wraps the frozen backend (gen.generate_dataset, apply.whiten, apply.regularize_psd) without editing it.
One process generates one shard: one split, a block of injections followed by a block of noise-only samples.
The seeding scheme is in ../SEEDING.md; the dataset layout and decisions are in ml_practice_v2/GENERATOR_OVERHAUL.md.
"""

import hashlib
import inspect
import json
import logging
import os
import socket
import subprocess
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
import pycbc
import pycbc.detector
import pycbc.fft.backend_support
import torch

import apply
import gen
from train import PARAM_COLUMNS

MASTER_SEED = 3021456217  # provenance in ../SEEDING.md

SPLITS = {'training': 0, 'validation': 1, 'test': 2}
PURPOSE_RNG, PURPOSE_PYCBC, PURPOSE_SNR = 0, 1, 2

FFT_BACKEND = 'numpy'
EXPECTED_FFT_MODULE = 'pycbc.fft.npfft'

SCHEMA_VERSION = 1
LOW_FREQUENCY_CUTOFF = 20.  # the value gen.main() uses
DETECTORS = ('H1', 'L1')
SAMPLE_LENGTH = 4096
DEFAULT_COUNTS = (5000, 5000)  # injections, noise-only

# sha256 of the frozen backend; a mismatch stops the run (recorded 2026-10-22, reading the working tree)
EXPECTED_BACKEND_HASHES = {
    'gen.py': 'db483f0420f7d36976cc7eddada750038dba58fe6c92756c83078bcfee3a421a',
    'apply.whiten': 'c85e9e18be2eab89b7a7f2318ccee9e4e773904118c2f51e13e8dc55a284f230',
    'apply.regularize_psd': '369138ae797090d9f0a19c8ad1edc8aca3b20d0372416272f889146c7b0caf15',
}


def pin_fft_backend():
    """Use pycbc's numpy FFT backend: under the default FFTW backend the waveforms differ at ~1e-13 between
    processes (BUGS.md OBS-025), probably because FFTW plans differ; the numpy backend did not differ in the probe."""
    pycbc.fft.backend_support.set_backend([FFT_BACKEND])
    active = pycbc.fft.backend_support.get_backend().__name__
    if active != EXPECTED_FFT_MODULE:
        raise RuntimeError("FFT backend is %s, expected %s" % (active, EXPECTED_FFT_MODULE))
    return active


def seed_seq(dataset_id, split, shard, purpose):
    return np.random.SeedSequence(MASTER_SEED, spawn_key=(dataset_id, SPLITS[split], shard, purpose))


def backend_hashes():
    def sha(data):
        return hashlib.sha256(data).hexdigest()

    with open(gen.__file__, 'rb') as f:
        gen_hash = sha(f.read())
    return {'gen.py': gen_hash,
            'apply.whiten': sha(inspect.getsource(apply.whiten).encode()),
            'apply.regularize_psd': sha(inspect.getsource(apply.regularize_psd).encode())}


def check_backend():
    found = backend_hashes()
    changed = {k: (EXPECTED_BACKEND_HASHES[k], found[k]) for k in found if found[k] != EXPECTED_BACKEND_HASHES[k]}
    if changed:
        raise RuntimeError("Frozen backend changed (expected, found): %s" % json.dumps(changed, indent=2))
    return found


def git_info():
    here = os.path.dirname(os.path.abspath(__file__))

    def run(*cmd):
        return subprocess.run(('git', '-C', here) + cmd, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return run('rev-parse', 'HEAD'), bool(run('status', '--porcelain', '--untracked-files=no'))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown', True


def generate_shard(dataset_id, split, shard, counts):
    """Seed once, then call the unchanged backend once for the shard; returns the Dataset."""
    pin_fft_backend()
    rng_seq = seed_seq(dataset_id, split, shard, PURPOSE_RNG)
    pycbc_seq = seed_seq(dataset_id, split, shard, PURPOSE_PYCBC)

    detectors = [pycbc.detector.Detector(name) for name in DETECTORS]
    colored_sample_length = SAMPLE_LENGTH + gen.max_filter_length
    noise_getter = gen.GaussianNoiseGetter(fpath=None, detectors=detectors,
                                           sample_length=colored_sample_length,
                                           low_frequency_cutoff=LOW_FREQUENCY_CUTOFF,
                                           segment_duration=gen.segment_duration,
                                           max_filter_duration=gen.max_filter_duration)
    noise_getter_iter = iter(noise_getter)

    # Seed last, once, so nothing above can consume the streams (SEEDING.md, "Usage pattern").
    gen.np_gen = np.random.default_rng(rng_seq)
    np.random.seed(pycbc_seq.generate_state(4))

    return gen.generate_dataset(counts, detectors, low_frequency_cutoff=LOW_FREQUENCY_CUTOFF,
                                approximant='IMRPhenomD', noise_getter_iter=noise_getter_iter,
                                sample_length=SAMPLE_LENGTH)


def write_provenance(outfile, dataset_id, split, shard, counts, hashes):
    sha, dirty = git_info()
    keys = {p: list(seed_seq(dataset_id, split, shard, p).spawn_key)
            for p in (PURPOSE_RNG, PURPOSE_PYCBC, PURPOSE_SNR)}
    attrs = {
        'schema_version': SCHEMA_VERSION,
        'kind': 'raw_shard',
        'sample_length': SAMPLE_LENGTH,
        'sample_rate': int(1. / gen.delta_t),
        'detectors': json.dumps(DETECTORS),
        'approximant': 'IMRPhenomD',
        'noise_source': 'gaussian, aLIGOZeroDetHighPower',
        'low_frequency_cutoff': LOW_FREQUENCY_CUTOFF,
        'segment_duration': gen.segment_duration,
        'max_filter_length': gen.max_filter_length,
        'whitened': True,
        'fft_backend': pycbc.fft.backend_support.get_backend().__name__,
        'master_seed': MASTER_SEED,
        'dataset_id': dataset_id,
        'split': split,
        'shard': shard,
        'seed_keys': json.dumps(keys),
        'n_injections': counts[0],
        'n_noise_only': counts[1],
        'columns': json.dumps(PARAM_COLUMNS),
        'backend_hashes': json.dumps(hashes),
        'git_sha': sha,
        'git_dirty': dirty,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'pycbc': pycbc.__version__,
        'h5py': h5py.__version__,
        'torch': torch.__version__,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(),
        'argv': json.dumps(sys.argv),
    }
    for name, value in attrs.items():
        # str() because torch.__version__ is a str subclass, which h5py cannot store
        outfile.attrs[name] = str(value) if isinstance(value, str) else value


def main():
    parser = ArgumentParser(description="Generate one seeded shard of the regenerated dataset (raw gen.py layout).")
    parser.add_argument('--split', choices=list(SPLITS), required=True)
    parser.add_argument('--shard', type=int, required=True, help="Shard number within the split.")
    parser.add_argument('--dataset-id', type=int, default=0)
    parser.add_argument('--counts', type=int, nargs=2, default=list(DEFAULT_COUNTS),
                        metavar=('INJECTIONS', 'NOISE_ONLY'),
                        help="Default: %d %d. Other values are for the determinism check." % DEFAULT_COUNTS)
    parser.add_argument('--out-dir', type=str, required=True)
    parser.add_argument('--force', action='store_true', help="Overwrite an existing shard file.")
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=logging.INFO if args.verbose else logging.WARN)

    hashes = check_backend()
    counts = tuple(args.counts)

    final = os.path.join(args.out_dir, 'd%d_%s_%03d.hdf' % (args.dataset_id, args.split, args.shard))
    partial = final + '.partial'
    if os.path.exists(final) and not args.force:
        raise RuntimeError("Shard file exists: %s" % final)
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = generate_shard(args.dataset_id, args.split, args.shard, counts)

    with h5py.File(partial, 'w') as outfile:
        write_provenance(outfile, args.dataset_id, args.split, args.shard, counts, hashes)
        dataset.save(outfile, args.split)
        outfile.attrs['complete'] = True
    os.replace(partial, final)
    logging.info("Wrote %s", final)


if __name__ == '__main__':
    main()
