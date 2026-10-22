# Handoff (ml-gw-search)

Written 2026-10-19T22:10Z, on `thinkpad-t530`.
This repository is the MLGWSC-1 code of Zelenka, Brügmann and Ohme (see `README.md`), used here to regenerate the training dataset of the `ml_practice_v2` q_value project.
The working rules, the machines table, and the full state of that project are in `ml_practice_v2` (`CLAUDE.md`, `HANDOFF.md`, `BUGS.md`, `GENERATOR_OVERHAUL.md`); this file covers only what concerns this repository.

## Read first

| File | What it tells you |
|------|-------------------|
| `HANDOFF.md` (this file) | the rules for working here, the state, what is next |
| `SEEDING.md` | the seeding scheme for the regenerated dataset, this repository's own master seed, and the determinism check |
| `../ml_practice_v2/GENERATOR_OVERHAUL.md` | what the generator lacks (G-01 to G-24), the target output file (section 4), the decisions (section 5), and the plan (section 7) |
| `../ml_practice_v2/BUGS.md` | BUG-007 to 009 and OBS-014 to 022 are about this generator |
| `mlgwsc-1/gen_py_data_pipeline.md` | notes on what `gen.py` saves and how the repackaging step works (it diverges from the copy in `ml_practice_v2/src/gwml/`, OBS-017) |

## Rules for working here

- **The generator backend is frozen** because it is part of a published paper: `mlgwsc-1/gen.py`, `whiten` and `regularize_psd` in `mlgwsc-1/apply.py`, and the thesis scripts are not edited, and nothing in them is made configurable.
  The only change so far is a label-level one (below), which the user accepted because it leaves the backend untouched.
  New behavior goes in new files beside them, calling the existing functions unchanged.
- **Never push to `origin`:** it is the upstream author's GitHub repository.
  The `dev` branch exists only in this clone, so nothing here is backed up unless the user adds a remote of their own.
- **Do not run data generation on `thinkpad-t530`** (CPU-only, 8 CPUs); generation is heavy pycbc work for the workstation or the lab machine, and only with the user's go-ahead.
- **Do not load `combined_repackaged.hdf` wholesale** (1.9 GB, untracked); read its structure with `h5py` metadata only.
- **Commits:** ask the user first; the subject is the bare `[dev-minor-N]` label, followed by bullets that are one sentence each on one line, then the trailers (the format is in `ml_practice_v2/CLAUDE.md`, §11).
- **Environment:** the `.venv` in this repository has Python 3.10, pycbc 2.9.0, h5py 3.16, numpy 2.2.6, and torch (CPU); run Python as `.venv/bin/python -I` when reading untrusted data.

## State

- Branch `dev`; HEAD is `[dev-minor-3]`, which adds `SEEDING.md` and `HANDOFF.md`; the commit before it, `[dev-minor-2]`, reworded `PARAM_COLUMNS` in `mlgwsc-1/train.py` to `mchirp, q, inclination, coa_phase, polarization_angle, declination, ra, injection_time, merger_time`, matching `ml_practice_v2/src/gwml/heads_spec.py` (metadata only, BUG-008).
- Both commits are unpushed, and `origin` is the upstream author's repository, so nothing here is backed up.
- Uncommitted: none of the tracked files.
- Untracked and not committed: `re_package_hdf.py`, `mlgwsc-1/gen_py_data_pipeline.md`, `mlgwsc-1/__pycache__/`, and `combined_repackaged.hdf`.
- No driver, merge step, or repackaging script for the new dataset exists yet.

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

- The shape of the driver, the repackaging script, and the merge step is a design to be built step by step with the user's go-ahead.
- The determinism check in `SEEDING.md` has not been run.
- Whether to keep `re_package_hdf.py` and the pipeline note untracked, or to commit them (OBS-017).
