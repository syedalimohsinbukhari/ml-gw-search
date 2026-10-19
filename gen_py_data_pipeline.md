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