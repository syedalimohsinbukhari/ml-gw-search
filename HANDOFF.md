# Handoff (ml-gw-search)

Written 2026-10-19T22:10Z, on `thinkpad-t530`; State and Open updated 2026-10-22T06:52Z.
This repository is the MLGWSC-1 code of Zelenka, Brügmann and Ohme (see `README.md`), used here to regenerate the training dataset of the `ml_practice_v2` q_value project.
The working rules, the machines table, and the full state of that project are in `ml_practice_v2` (`CLAUDE.md`, `HANDOFF.md`, `BUGS.md`, `GENERATOR_OVERHAUL.md`); this file covers only what concerns this repository.

## Read first

| File | What it tells you |
|------|-------------------|
| `HANDOFF.md` (this file) | the rules for working here, the state, what is next |
| `SEEDING.md` | the seeding scheme for the regenerated dataset, this repository's own master seed, and the determinism check |
| `../ml_practice_v2/GENERATOR_OVERHAUL.md` | what the generator lacks (G-01 to G-24), the target output file (section 4), the decisions (section 5), and the plan (section 7) |
| `../ml_practice_v2/BUGS.md` | BUG-007 to 009 and OBS-014 to 022 are about this generator |
| `gen_py_data_pipeline.md` (repository root; `mlgwsc-1/gen_py_data_pipeline.md` on `thinkpad-t530`) | notes on what `gen.py` saves and how the repackaging step works (it diverges from the copy in `ml_practice_v2/src/gwml/`, OBS-017) |

## Rules for working here

- **The generator backend is frozen** because it is part of a published paper: `mlgwsc-1/gen.py`, `whiten` and `regularize_psd` in `mlgwsc-1/apply.py`, and the thesis scripts are not edited, and nothing in them is made configurable.
  The only change so far is a label-level one (below), which the user accepted because it leaves the backend untouched.
  New behavior goes in new files beside them, calling the existing functions unchanged.
- **Remotes (reconfigured on `sarlws5-Precision-7820-Tower`, 2026-10-22T05:47Z):** `origin` is the user's fork (`syedalimohsinbukhari/ml-gw-search`), and `dev` and `main` track it.
  `upstream` is the original author's repository (`ondrzel/ml-gw-search`); it is fetch-only, with its push URL set to `DISABLED`, and nothing is ever pushed to it.
  Ask before pushing to `origin`; a yes covers that push only.
  `thinkpad-t530` was not reconfigured and may still have the old layout (`origin` = the author's repository), so check `git remote -v` there before pushing.
- **Do not run data generation on `thinkpad-t530`** (CPU-only, 8 CPUs); generation is heavy pycbc work for the workstation or the lab machine, and only with the user's go-ahead.
- **Do not load `combined_repackaged.hdf` wholesale** (1.9 GB, untracked); read its structure with `h5py` metadata only.
- **Commits:** ask the user first; the subject is the bare `[dev-minor-N]` label, followed by bullets that are one sentence each on one line, then the trailers (the format is in `ml_practice_v2/CLAUDE.md`, §11).
- **Environment:** the `.venv` in this repository has Python 3.10, pycbc 2.9.0, h5py 3.16, numpy 2.2.6, and torch (CPU); run Python as `.venv/bin/python -I` when reading untrusted data.

## State

Updated 2026-10-22T06:52Z on `sarlws5-Precision-7820-Tower`.

- Branch `dev`, tracking `origin/dev` (the user's fork); the history is `[dev-minor-4]` (a merge of the fork's and this machine's diverged histories), `[dev-minor-5]` (the remote-layout update of this file), the user's `[dev-minor-5]` and `[dev-minor-5-1]` (the pipeline note under `mlgwsc-1/` and a `.gitignore` line for `combined_repackaged.hdf`, pulled by a rebase), and then `[dev-minor-6]`, `[dev-minor-7]`, and `[dev-minor-8]` (below); `[dev-minor-7]` and `[dev-minor-8]` are committed locally and not yet pushed.
- The fork's earlier `[dev-minor-2]` reworded `PARAM_COLUMNS` in `mlgwsc-1/train.py` to `mchirp, q, inclination, coa_phase, polarization_angle, declination, ra, injection_time, merger_time`, matching `ml_practice_v2/src/gwml/heads_spec.py` (metadata only, BUG-008, now `VERIFIED` in a written shard).
- Two copies of the pipeline note now exist, `gen_py_data_pipeline.md` at the root and `mlgwsc-1/gen_py_data_pipeline.md`, and they differ (OBS-017).
- `[dev-minor-6]` adds the seeded driver and the determinism check, both in `mlgwsc-1/`:
  - `gen_seeded.py` generates one shard (`--split`, `--shard`, `--dataset-id`, `--counts`, `--out-dir`) in the raw `gen.py` layout, as `d<dataset_id>_<split>_<shard>.hdf`, written under a `.partial` name and renamed on success.
  - It holds `MASTER_SEED`, seeds once per shard, refuses to run if the sha256 of `gen.py`, `apply.whiten`, or `apply.regularize_psd` differs from the pinned values, pins pycbc's `numpy` FFT backend (BUGS.md OBS-025 in `ml_practice_v2`), and writes the provenance attributes listed in `SEEDING.md`.
  - `check_determinism.py` covers all six steps of the check in `SEEDING.md`: steps 1 to 4 for the driver, step 5 for `re_package_hdf_with_noise.py` (two runs identical, SNRs in range and different for another shard or split, strain = noise + snr × unit waveform to float32 rounding), step 6 for `merge_hdf_shards.py` (two input orders give identical arrays, equal to the shard-order concatenation).
- Result (2026-10-22, three runs, 10 injections + 5 noise-only per shard): 9 of 9 checks passed each time with the `numpy` FFT backend.
  Under the FFTW default, step 1 failed by about 1e-13 in one waveform element (OBS-025), which is why the backend is pinned.
  Not yet checked at shard scale (5,000 injections): queued in `ml_practice_v2/VERIFY.md`.
- Untracked or ignored: `combined_repackaged.hdf`, `out.hdf`, and `__pycache__/` are ignored; the scratch outputs of the check are in the session scratchpad, not in this repository.
- `[dev-minor-7]` adds `re_package_hdf_with_noise.py` (repo root; `re_package_hdf.py` is unchanged):
  - It repackages one raw shard into `d<id>_<split>_<shard>_repackaged.hdf`, with `h1`, `l1` (noise + snr × unit waveform), `h1_unit_waveform`, `l1_unit_waveform`, `params` (the nine generator columns then `snr`, float64), and the noise-only rows in `noise_h1`, `noise_l1`.
  - The SNR is uniform on (7, 15), drawn once per shard from the purpose-2 key; it copies the generation provenance and adds the source sha256, the SNR range and key, and the repackaging git SHA, time, and host.
  - Checked on the 10-injection scratch shard: the strain equals an independent recomputation to float32 rounding (a wrong-SNR control fails), noise-only rows and parameters are copied exactly, two runs are identical, and the loader's three reads give the expected shapes and dtypes; the error paths and a full-size shard are untested.
  - The unit waveforms are named `*_unit_waveform` because the old `h1_waveform` held the scaled waveform; nothing reads them today.
- `[dev-minor-8]` adds `merge_hdf_shards.py` (repo root):
  - It sorts shards by split and shard number, checks that shard numbers are contiguous and that the generating constants and library versions match, copies shard by shard into pre-sized datasets, and writes under `.partial` with `complete` last; `--expect training=10 validation=2 test=1` checks the counts.
  - Checked ad hoc on synthetic shards copied from the one real repackaged shard: three input orders give identical arrays equal to an independent concatenation, eight failure cases raise and leave nothing behind, and the command line matches the Python call.
  - Untested: the full merge (about 6.4 GB over 13 shards: memory and time unknown), real shards from two processes, and an interrupted write.
- Result (2026-10-22T07:26Z, three runs of `check_determinism.py --counts 10 5`, shards generated by fresh processes): 17 of 17 checks passed each time, steps 1 to 6.
  The contract check in `ml_practice_v2` (`scripts/check_contract.py`) passed 68 of 68 on a merged file of those real shards, and `--full` failed it only on the D-03 counts and the missing test split, as it should.
- Not done: a full-size shard, the 13-shard generation, and the full merge.

## The plan, in short (details in `ml_practice_v2/GENERATOR_OVERHAUL.md`)

- The dataset: `IMRPhenomD`, spinless, legacy priors, Gaussian aLIGO design-PSD noise, H1 and L1, 4096-sample windows at 2048 Hz.
- The counts: training 100,000, validation 20,000, test 10,000, half of each noise-only, generated as 13 shards of 10,000 (5,000 injections and 5,000 noise-only each), one process per shard, then merged.
- The layout: injection rows in `h1`, `l1`, `params` (so the consumer's loader is unchanged), noise-only rows in `noise_h1`, `noise_l1`; the strain plus the unit-SNR waveform are stored, and one SNR per sample in (7, 15) is baked in and recorded as a column.
- The new files to write beside the frozen backend:
  - a seeded driver that rebinds `gen.np_gen`, seeds `numpy.random` once per shard, and calls `generate_dataset` unchanged;
  - `re_package_hdf_with_noise.py` (the original `re_package_hdf.py` stays unchanged);
  - a merge step that writes under a temporary name, sorts by split and shard, and sets a final `complete` attribute;
  - pinned hashes of the frozen backend, recorded in every generated file with the keys, arguments, library versions, git SHA, time, and host (`SEEDING.md`, "What each generated file records").
- Known and accepted: `injection_time` stays float32 (128 s resolution) because the cast is inside the backend (BUG-009); the torch coupling of `gen.py` (it imports `Dataset` from `train.py`) stays.

## Open

- The determinism of a full 5,000-injection shard, the time one takes, and the cost of the full merge are unknown (`ml_practice_v2/VERIFY.md`, OBS-025).
- Whether the two copies of the pipeline note are reconciled or one is dropped (OBS-017).
