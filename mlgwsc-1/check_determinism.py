#! /usr/bin/env python
"""Determinism check for the generator chain, the six steps of ../SEEDING.md ("Determinism check").

Steps 1 to 4 test gen_seeded.py, step 5 re_package_hdf_with_noise.py, step 6 merge_hdf_shards.py.
Runs a shrunken workload (default 3 injections + 3 noise-only per shard) and writes only under --work-dir.
Generation is CPU-heavy pycbc work: run it only with the user's go-ahead, and not on thinkpad-t530.

Falsified by: any difference in steps 1, 3, 5 (two repackaging runs) or 6, any equality in steps 2, 4, or 5 (SNRs of
another shard or split), an SNR outside the range, or a strain that is not noise + snr * unit waveform.
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

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.join(HERE, 'gen_seeded.py')
REPACKAGER = os.path.join(os.path.dirname(HERE), 're_package_hdf_with_noise.py')
MERGER = os.path.join(os.path.dirname(HERE), 'merge_hdf_shards.py')
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


def run_repackage(shard_path, out_dir):
    subprocess.run([sys.executable, REPACKAGER, '--input', shard_path, '--out-dir', out_dir, '--force'], check=True)
    return os.path.join(out_dir, os.path.basename(shard_path).replace('.hdf', '_repackaged.hdf'))


def run_merge(inputs, output):
    subprocess.run([sys.executable, MERGER, '--inputs', *inputs, '--output', output, '--force'], check=True)
    return output


def all_arrays(path):
    with h5py.File(path, 'r') as f:
        assert f.attrs['complete'], "file not marked complete: %s" % path
        return {'%s/%s' % (split, name): f[split][name][()] for split in f for name in f[split]}


def same_arrays(a, b):
    return a.keys() == b.keys() and all(a[k].shape == b[k].shape and np.array_equal(a[k], b[k]) for k in a)


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
    t0_path = run_cli(run1, 'training', 0, counts)
    a1 = arrays_from_file(t0_path, 'training')
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

    # Steps 5 and 6 need real shards from fresh processes: training 1 and validation 0 beside training 0.
    t1_path = run_cli(os.path.join(args.work_dir, 'run3'), 'training', 1, counts)
    v0_path = run_cli(os.path.join(args.work_dir, 'run4'), 'validation', 0, counts)

    # Step 5: repackaging the same shard twice gives identical SNRs and arrays; other shards and splits differ.
    rp_a = run_repackage(t0_path, os.path.join(args.work_dir, 'rp_a'))
    rp_b = run_repackage(t0_path, os.path.join(args.work_dir, 'rp_b'))
    rp_t1 = run_repackage(t1_path, os.path.join(args.work_dir, 'rp_t1'))
    rp_v0 = run_repackage(v0_path, os.path.join(args.work_dir, 'rp_v0'))
    ra, rb, rt1, rv0 = (all_arrays(p) for p in (rp_a, rp_b, rp_t1, rp_v0))
    record(5, 'repackaging one shard twice gives identical arrays', same_arrays(ra, rb))
    snr = ra['training/params'][:, -1]
    record(5, 'baked SNRs lie in [7, 15]', bool(np.all((snr >= 7.) & (snr <= 15.))),
           'min %.3f max %.3f' % (snr.min(), snr.max()))
    record(5, 'SNRs differ for another shard number',
           not np.array_equal(snr, rt1['training/params'][:, -1]))
    record(5, 'SNRs differ for the same shard number in another split',
           not np.array_equal(snr, rv0['validation/params'][:, -1]))
    src_noise = gen_a['noises'][:counts[0]]
    derived = ra['training/h1'].astype(np.float64) - snr[:, None] * ra['training/h1_unit_waveform']
    tol = 64 * np.finfo(np.float32).eps * max(1., float(np.abs(src_noise[:, 0]).max()))
    err = float(np.abs(derived - src_noise[:, 0]).max())
    record(5, 'strain minus snr * unit waveform equals the source noise to float32 rounding', err <= tol,
           'max error %.3g, tolerance %.3g' % (err, tol))

    # Step 6: merging the shards in two input orders gives identical arrays, equal to the shard-order concatenation.
    m1 = all_arrays(run_merge([rp_a, rp_t1, rp_v0], os.path.join(args.work_dir, 'merge_fwd.hdf')))
    m2 = all_arrays(run_merge([rp_v0, rp_t1, rp_a], os.path.join(args.work_dir, 'merge_rev.hdf')))
    record(6, 'two input orders give identical merged arrays', same_arrays(m1, m2))
    expected = {'training/%s' % k.split('/')[1]: np.concatenate([ra[k], rt1[k]]) for k in ra}
    record(6, 'merged training rows are shard 0 then shard 1',
           all(np.array_equal(m1[k], v) for k, v in expected.items()))
    record(6, 'merged validation rows equal the validation shard',
           all(np.array_equal(m1[k], v) for k, v in rv0.items()))

    failed = [r for r in results if not r[2]]
    print('\n%d of %d checks passed (steps 1 to 6)' % (len(results) - len(failed), len(results)))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
