"""Created on Jun 10 13:20:04 2026"""

import h5py as h5
import numpy as np


def package_split(noises, waveforms, waveform_params, rng_engine, snr_range=(7, 15)):
    assert len(noises) == len(waveforms) == len(waveform_params), (
        f"Shape mismatch: noises={len(noises)}, waveforms={len(waveforms)}, "
        f"waveform_params={len(waveform_params)}"
    )
    h1, l1, h1_wv, l1_wv, params = [], [], [], [], []
    for noise, wvf, par in zip(noises, waveforms, waveform_params):
        snr = rng_engine.uniform(*snr_range)
        _scaled_h1 = snr * wvf[0]
        h1.append(noise[0] + _scaled_h1)
        h1_wv.append(_scaled_h1)

        _scaled_l1 = snr * wvf[1]
        l1.append(noise[1] + _scaled_l1)
        l1_wv.append(_scaled_l1)

        par = np.append(par, snr)

        params.append(par)
    return np.array(h1), np.array(l1), np.array(h1_wv), np.array(l1_wv), np.array(params)


def write_split(group, h1, l1, h1_wv, l1_wv, params):
    group.create_dataset("h1", data=h1)
    group.create_dataset("l1", data=l1)
    group.create_dataset("h1_waveform", data=h1_wv)
    group.create_dataset("l1_waveform", data=l1_wv)
    group.create_dataset("params", data=params)


rng = np.random.default_rng(132354)

with (
    h5.File(name="./out.hdf", mode="r") as f_in,
    h5.File(name="out_repackaged.hdf", mode="w") as f_out
):
    for split in ("training", "validation"):
        grp = f_in[split]
        wv = grp["waveforms"][...]
        n_wv = wv.shape[0]
        arrays = package_split(grp["noises"][...][:n_wv], wv,
                               grp["waveform_params"][...], rng)
        write_split(f_out.create_group(split), *arrays)
        h1, l1, h1_wv, l1_wv, params = arrays
        print(f"{split}: h1={h1.shape}, l1={l1.shape}, params={params.shape}")
