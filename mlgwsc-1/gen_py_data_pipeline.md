# `gen.py` data pipeline notes

## What gets saved

`gen.py` generates and saves **noise and waveforms as two separate arrays** — they are never summed inside `gen.py`.

- `generate_dataset()` builds three arrays: `noises`, `waveforms`, `waveform_params`.
- `main()` wraps them in a `Dataset` (from `train.py`) and calls `TrainDS.save(outfile, 'training')` / `ValidDS.save(outfile, 'validation')` (gen.py:348, 355).
- `Dataset.save()` (train.py:89-98) writes `noises` and `waveforms` as two separate HDF5 datasets under a group. No addition happens here.

## Whitening: default options (Gaussian noise)

With default args (`--real-noise-file` not passed → `args.real_noise_file is None`), `main()` picks `GaussianNoiseGetter` (gen.py:331-337):

```python
if args.real_noise_file is None:
    noise_getter = GaussianNoiseGetter(...)
else:
    noise_getter = HDFNoiseGetter(args.real_noise_file, detectors=detectors)
```

- **Noise**: `GaussianNoiseGetter.__next__` (gen.py:109-118) generates colored Gaussian noise from a PSD and explicitly whitens it before returning.
- **Waveform**: in `generate_dataset`, the projected waveform is rescaled by its own optimal network SNR (`waveform = ... / network_snr`, gen.py:240 — making it effectively a *unit-SNR* template), then explicitly whitened (gen.py:242-245).

So with default options, **both arrays stored in the output file are already whitened.**

## Caveat: real-noise path (`--real-noise-file`)

If `--real-noise-file` is passed, `HDFNoiseGetter` is used instead. Its `__next__` (gen.py:73-79) just returns raw noise slices straight from the file — these come from `slice_real_noise.py`, whose `SegmentSlicer` is called with `white=False`, i.e. **unwhitened, colored** real noise. `gen.py` never whitens this noise before storing it (no call to `whiten()` on the real-noise path). This means with `--real-noise-file`, you'd end up with a whitened waveform being combined against colored (unwhitened) real noise — a mismatch worth being aware of if you use that option.

## Where noise + waveform actually get combined

Never inside `gen.py`. The combination happens later, at read time, in `train.py`'s `Dataset.__getitem__` (train.py:71-79):

```python
def __getitem__(self, i):
    if i < len(self.waveforms):
        snr = self.rng.uniform(*self.snr_range)   # default snr_range = (5., 15.)
        data = self.noises[i] + snr * self.waveforms[i]
        params = self.waveform_params[i] if self.waveform_params is not None else self._null_params
        return data, self.wave_label, params
    else:
        return self.noises[i], self.noise_label, self._null_params
```

Notes:
- `snr` is **not** stored anywhere — it's drawn fresh, uniformly from `snr_range`, every time `__getitem__` is called. So the same `(noise[i], waveform[i])` pair can produce a different combined sample each epoch.
- Indices `i >= num_waveforms` (the "pure noise" half of the dataset) have no waveform to add and are returned as noise-only samples labeled with `noise_label`.

## Practical implication

With default (Gaussian noise) options, since both `noises[i]` and `waveforms[i]` are already whitened when saved, you can manually reproduce a valid training sample outside of `train.py` via:

```python
sample = noises[i] + snr * waveforms[i]   # snr in [5, 15] by default
```

This is exactly what the training loop does on the fly — it's not something `gen.py` does or needs to do itself.

## Post-processing: `re_package_hdf.py`

`re_package_hdf.py` is a standalone script (not invoked by `gen.py`/`train.py`) that reads a `gen.py` output file (hardcoded path `./out.hdf`) and rewrites it into a new file (hardcoded `out_repackaged.hdf`) with the noise+waveform combination already baked in, instead of leaving it to `Dataset.__getitem__` at train time.

For each of the `training`/`validation` groups:

- Reads `waveforms`, `noises`, `waveform_params` from the input group. **Truncates `noises` to `noises[:n_wv]`**, where `n_wv = waveforms.shape[0]` — i.e. it keeps only the noise slices that pair with an injection and **silently discards the pure-noise half of the dataset** (the `num_noises` samples gen.py also generated, indices `>= num_waveforms`). The repackaged file has no noise-only class.
- For every `(noise, waveform, params)` triple, `package_split()` (re_package_hdf.py:12-30):
  - draws one `snr` per sample from `np.random.default_rng(132354).uniform(*snr_range)`, default `snr_range=(7, 15)` — note this differs from `train.py`'s default `(5., 15.)`, and this file uses a **fixed seed**, so the SNR-per-sample is baked in and reproducible, unlike `train.py` where a fresh SNR is redrawn every epoch (see above).
  - handles **any number of detector channels** — `detector_names(n_detectors)` (re_package_hdf.py:5-9) names them `h1, l1, v1, k1` (same order convention as gen.py's `detectors_abbr = ('H1','L1','V1','K1')[:args.detectors]`) for up to 4 detectors, falling back to generic `det0, det1, ...` beyond that. `n_detectors` is read from `waveforms.shape[1]`, so it adapts automatically to however many detectors the source `out.hdf` was generated with (`gen.py --detectors N`) instead of assuming 2.
  - computes `scaled_wv = snr * waveform[det]` and `combined = noise[det] + scaled_wv` per detector — same additive formula as `Dataset.__getitem__`.
  - appends `snr` as a 10th column onto the original 9-column `waveform_params` row (`par = np.append(par, snr)`).

`write_split()` (re_package_hdf.py:33-38) then writes, per group, `2*n_detectors + 1` flat datasets instead of gen.py's three:

| dataset | shape | contents |
|---|---|---|
| `{det}` (e.g. `h1`, `l1`, ...) | `(n_wv, sample_length)` | that detector's noise + scaled waveform (the noisy training sample), one dataset per detector |
| `{det}_waveform` (e.g. `h1_waveform`, ...) | `(n_wv, sample_length)` | that detector's waveform alone, pre-scaled by `snr` (no noise) |
| `params` | `(n_wv, 10)` | original `waveform_params` columns + appended `snr` |

Since `{det}_waveform` is already scaled by `snr` at write time (`scaled = snr * wvf[det_idx]`, re_package_hdf.py:26-27), recovering the pure noise for a given detector/sample from the repackaged file needs no `snr` or original `waveforms` lookup — it's just:

```python
noise = f[split][det][i] - f[split][f"{det}_waveform"][i]   # e.g. det = "h1"
```

Caveats vs. the original `gen.py` output:
- No pure-noise samples survive — only injection samples remain, so any downstream consumer expecting a two-class (noise vs. signal) dataset from this file needs a different noise source or a different repackaging strategy.
- The `columns` attrs that `train.py`'s `Dataset.save()` attaches to `waveform_params` (`PARAM_COLUMNS`) are **not** carried over or updated for the new 10th (`snr`) column — `params` in the repackaged file is a plain, unlabeled array.
- Top-level file attrs from `gen.py` (`sample_length`, `sample_rate`) are not copied to `out_repackaged.hdf`.
- H1 and L1 are split into separate datasets/keys rather than kept as a single `(detectors, sample_length)` array per sample, so any loader for this file needs to know to stack `h1`/`l1` back together (or consume them as separate channels) — it's not a drop-in replacement for `train.py`'s `Dataset.load()`, which still expects the original `noises`/`waveforms` group layout.