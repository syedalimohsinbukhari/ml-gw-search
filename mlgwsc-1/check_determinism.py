#! /usr/bin/env python
"""Determinism check for gen_seeded.py, steps 1 to 4 of ../SEEDING.md ("Determinism check").

Steps 5 (repackaging SNRs) and 6 (merge order) need scripts that do not exist yet and are reported as NOT RUN.
Runs a shrunken workload (default 3 injections + 3 noise-only per shard) and writes only under --work-dir.
Generation is CPU-heavy pycbc work: run it only with the user's go-ahead, and not on thinkpad-t530.

Falsified by: any difference in steps 1 or 3, any equality in steps 2 or 4.
Arrays are compared, not file bytes.
A difference at float-rounding level is reported as a FAIL with its size; it is a decision point for the user.
"""

import os
import subprocess
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np

import gen_seeded

DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gen_seeded.py')
ARRAYS = ('noises', 'waveforms', 'waveform_params')

results = []


def record(step, name, ok, detail=''):
    results.append((step, name, ok))
    print('%s  step %s: %s%s' % ('PASS' if ok else 'FAIL', step, name, ('  [%s]' % detail) if detail else ''))


def arrays_of(dataset):
    return {'noises': dataset.noises.numpy(), 'waveforms': dataset.waveforms.numpy(),
            'waveform_params': dataset.waveform_params.numpy()}


def arrays_from_file(path, split):
    with h5py.File(path, 'r') as f:
        assert f.attrs['complete'], "file not marked complete: %s" % path
        return {k: f[split][k][()] for k in ARRAYS}


def equal(a, b):
    return all(a[k].shape == b[k].shape and np.array_equal(a[k], b[k]) for k in ARRAYS)


def max_diff(a, b):
    return max(float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) if a[k].shape == b[k].shape
               else float('inf') for k in ARRAYS)


def run_cli(out_dir, split, shard, counts):
    """One fresh process per call, so no state can be shared between the two runs of step 1."""
    cmd = [sys.executable, DRIVER, '--split', split, '--shard', str(shard), '--out-dir', out_dir, '--force',
           '--counts', str(counts[0]), str(counts[1])]
    subprocess.run(cmd, check=True)
    return os.path.join(out_dir, 'd0_%s_%03d.hdf' % (split, shard))


def main():
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--work-dir', required=True, help="Scratch directory for the two step-1 shard files.")
    parser.add_argument('--counts', type=int, nargs=2, default=[3, 3], metavar=('INJECTIONS', 'NOISE_ONLY'))
    args = parser.parse_args()
    counts = tuple(args.counts)
    gen_seeded.check_backend()

    # Step 1: the same key, two fresh processes, through the real command line and file writer.
    run1 = os.path.join(args.work_dir, 'run1')
    run2 = os.path.join(args.work_dir, 'run2')
    a1 = arrays_from_file(run_cli(run1, 'training', 0, counts), 'training')
    a2 = arrays_from_file(run_cli(run2, 'training', 0, counts), 'training')
    ok = equal(a1, a2)
    record(1, 'same key in two processes gives identical noises, waveforms, params', ok,
           '' if ok else 'max abs difference %.3g' % max_diff(a1, a2))
    with h5py.File(os.path.join(run1, 'd0_training_000.hdf'), 'r') as f, \
            h5py.File(os.path.join(run2, 'd0_training_000.hdf'), 'r') as g:
        skip = {'generated_utc', 'argv'}
        diff = sorted(k for k in f.attrs if k not in skip and not np.array_equal(f.attrs[k], g.attrs[k]))
        record(1, 'provenance attributes identical apart from time and argv', not diff, ', '.join(diff))

    # Steps 2 and 3 in one process: A, B (next shard), C (same shard number, another split), then A again.
    gen_a = arrays_of(gen_seeded.generate_shard(0, 'training', 0, counts))
    gen_b = arrays_of(gen_seeded.generate_shard(0, 'training', 1, counts))
    gen_c = arrays_of(gen_seeded.generate_shard(0, 'validation', 0, counts))
    gen_a_again = arrays_of(gen_seeded.generate_shard(0, 'training', 0, counts))

    record(2, 'another shard number differs in noises', not np.array_equal(gen_a['noises'], gen_b['noises']))
    record(2, 'another shard number differs in waveform_params',
           not np.array_equal(gen_a['waveform_params'], gen_b['waveform_params']))
    record(2, 'same shard number in another split differs in noises',
           not np.array_equal(gen_a['noises'], gen_c['noises']))
    record(2, 'same shard number in another split differs in waveform_params',
           not np.array_equal(gen_a['waveform_params'], gen_c['waveform_params']))

    ok = equal(gen_a, gen_a_again)
    record(3, 'A, B, C, then A again in one process: the two copies of A are identical', ok,
           '' if ok else 'max abs difference %.3g' % max_diff(gen_a, gen_a_again))
    ok = equal(gen_a, a1)
    record(3, 'in-process A equals the file written by the command line', ok,
           '' if ok else 'max abs difference %.3g' % max_diff(gen_a, a1))

    # Step 4: H1 and L1 noise differ within every sample (noises has shape (N, detectors, length)).
    noises = gen_a['noises']
    same = [i for i in range(len(noises)) if np.array_equal(noises[i, 0], noises[i, 1])]
    record(4, 'H1 and L1 noise differ within every sample', not same, 'identical samples: %s' % same if same else '')

    print('NOT RUN  step 5: repackaging SNRs (re_package_hdf_with_noise.py does not exist yet)')
    print('NOT RUN  step 6: merge order (merge step does not exist yet)')

    failed = [r for r in results if not r[2]]
    print('\n%d of %d checks passed (steps 1 to 4 of 6)' % (len(results) - len(failed), len(results)))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
