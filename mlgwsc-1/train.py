#! /usr/bin/env python

# Copyright 2022 Ondřej Zelenka

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import logging
import os
import os.path
### Import modules
from argparse import ArgumentParser

import h5py
import numpy as np
import torch
from tqdm import tqdm

from apply import dtype
from apply import get_coherent_network, get_coincident_network, get_base_network2

### Column names for regression parameter labels
PARAM_COLUMNS = [
    'mass1',  # M_sun,  range [10, 50]
    'mass2',  # M_sun,  range [10, mass1]
    'inclination',  # rad,    range [0, 2pi]
    'coa_phase',  # rad,    range [0, 2pi]
    'pol_angle',  # rad,    range [0, 2pi]
    'declination',  # rad,    range [-pi/2, pi/2]
    'right_ascension',  # rad,    range [0, 2pi]
    'merger_time_gps',  # GPS seconds (O3a era, provenance only)
    'merger_time_in_window',  # seconds from start of whitened window to coalescence
]
NUM_PARAMS = len(PARAM_COLUMNS)


### Basic dataset class for easy PyTorch loading
class Dataset(torch.utils.data.Dataset):
    def __init__(self, noises=None, waveforms=None, waveform_params=None,
                 store_device='cpu', train_device='cpu',
                 snr_range=(5., 15.)):
        torch.utils.data.Dataset.__init__(self)
        self.noises = noises
        self.waveforms = waveforms
        self.waveform_params = waveform_params
        self.store_device = store_device
        self.train_device = train_device  # kept for compat; device transfers happen in train loop
        if not self.noises is None:
            self.convert()
        self.rng = np.random.default_rng()
        self.snr_range = snr_range
        # Always keep labels/null params on CPU — batches are moved to train_device in the loop
        self.wave_label = torch.tensor([1., 0.], dtype=dtype)
        self.noise_label = torch.tensor([0., 1.], dtype=dtype)
        self._null_params = torch.zeros(NUM_PARAMS, dtype=dtype)
        return

    def __len__(self):
        return len(self.noises)

    def __getitem__(self, i):
        if i < len(self.waveforms):
            snr = self.rng.uniform(*self.snr_range)
            # Return on store_device — the training loop moves the whole batch at once
            data = self.noises[i] + snr * self.waveforms[i]
            params = self.waveform_params[i] if self.waveform_params is not None else self._null_params
            return data, self.wave_label, params
        else:
            return self.noises[i], self.noise_label, self._null_params

    def convert(self):
        self.noises = torch.from_numpy(self.noises).to(dtype=dtype, device=self.store_device)
        self.waveforms = torch.from_numpy(self.waveforms).to(dtype=dtype, device=self.store_device)
        if self.waveform_params is not None:
            self.waveform_params = torch.from_numpy(
                self.waveform_params.astype(np.float32)
            ).to(dtype=dtype, device=self.store_device)

    def save(self, h5py_file, group_name):
        if group_name in h5py_file.keys():
            raise IOError("Group '%s' in file already exists." % group_name)
        new_group = h5py_file.create_group(group_name)
        new_group.create_dataset('waveforms', data=self.waveforms.cpu().numpy())
        new_group.create_dataset('noises', data=self.noises.cpu().numpy())
        if self.waveform_params is not None:
            ds = new_group.create_dataset('waveform_params',
                                          data=self.waveform_params.cpu().numpy())
            ds.attrs['columns'] = PARAM_COLUMNS

    def load(self, h5py_file, group_name):
        if group_name not in h5py_file.keys():
            raise IOError("Group '%s' in file doesn't exist." % group_name)
        group = h5py_file[group_name]
        self.noises = group['noises'][()]
        self.waveforms = group['waveforms'][()]
        if 'waveform_params' in group:
            self.waveform_params = group['waveform_params'][()]
        else:
            self.waveform_params = None  # backward compat: old files without params
        self.convert()


### Basic dataset class for easy PyTorch loading
### Outdated class with fixed injections
# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, samples=None, labels=None,
#                 store_device='cpu', train_device='cpu'):
#         torch.utils.data.Dataset.__init__(self)
#         self.samples = samples
#         self.labels = labels
#         self.store_device = store_device
#         self.train_device = train_device
#         if not self.samples is None:
#             self.convert()
#         return

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, i):
#         sample = self.samples[i].to(device=self.train_device)
#         label = self.labels[i].to(device=self.train_device)
#         return sample, label

#     def convert(self):
#         self.samples = torch.from_numpy(self.samples).to(dtype=dtype, device=self.store_device)
#         self.labels = torch.from_numpy(self.labels).to(dtype=dtype, device=self.store_device)
#         assert len(self.samples)==len(self.labels)

#     def save(self, h5py_file, group_name):
#         if group_name in h5py_file.keys():
#             raise IOError("Group '%s' in file already exists." % group_name)
#         else:
#             new_group = h5py_file.create_group(group_name)
#             new_group.create_dataset('samples', data=self.samples.cpu().numpy())
#             new_group.create_dataset('labels', data=self.labels.cpu().numpy())

#     def load(self, h5py_file, group_name):
#         if group_name in h5py_file.keys():
#             group = h5py_file[group_name]
#             self.samples = group['samples'][()]
#             self.labels = group['labels'][()]
#             self.convert()
#         else:
#             raise IOError("Group '%s' in file doesn't exist." % group_name)


class reg_BCELoss(torch.nn.BCELoss):
    def __init__(self, *args, epsilon=1e-6, dim=None, **kwargs):
        torch.nn.BCELoss.__init__(self, *args, **kwargs)
        assert isinstance(dim, int)
        self.regularization_dim = dim
        self.regularization_A = epsilon
        self.regularization_B = 1. - epsilon * self.regularization_dim

    def forward(self, inputs, target, *args, **kwargs):
        assert inputs.shape[-1] == self.regularization_dim
        transformed_input = self.regularization_A + self.regularization_B * inputs
        return torch.nn.BCELoss.forward(self, transformed_input, target, *args, **kwargs)


def train(Network, training_dataset, validation_dataset, output_training,
          batch_size=32, learning_rate=5e-5, epochs=100,
          clip_norm=100, verbose=False, force=False,
          train_device='cpu', num_workers=0):
    """Train a network on given data.
    
    Arguments
    ---------
    Network : network as returned by get_network
        The network to train.
    training_dataset : (np.array, np.array)
        The data to use for training. The first entry has to contain the
        input data, whereas the second entry has to contain the target
        labels.
    validation_dataset : (np.array, np.array)
        The data to use for validation. The first entry has to contain
        the input data, whereas the second entry has to contain the
        target labels.
    output_training : str
        Path to a directory in which the loss history and the best
        network weights will be stored.
    weights_path: str
        Path where the trained network weights will be stored.
    batch_size : {int, 32}
        The mini-batch size used for training the network.
    learning_rate : {float, 5e-5}
        The learning rate to use with the optimizer.
    epochs : {int, 100}
        The number of full passes over the training data.
    clip_norm : {float, 100}
        The value at which to clip the gradient to prevent exploding
        gradients.
    verbose : {bool, False}
        Print update messages.
    force : {bool, False}
        Overwrite existing output files.
    train_device : {str, 'cpu'}
        Device to move each batch to before the forward pass.
    num_workers : {int, 0}
        Number of DataLoader worker processes for prefetching.
        Use 0 if store_device is CUDA.

    Returns
    -------
    network
    """
    ### Set up data loaders as a PyTorch convenience
    logging.debug("Setting up datasets and data loaders.")
    pin = num_workers > 0 and 'cuda' in str(train_device)
    persistent = num_workers > 0
    TrainDL = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin,
                                          persistent_workers=persistent)
    ValidDL = torch.utils.data.DataLoader(validation_dataset, batch_size=500, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin,
                                          persistent_workers=persistent)

    ### Initialize loss function, optimizer and output file
    logging.debug("Initializing loss function, optimizer and output file.")
    loss = reg_BCELoss(dim=2)
    opt = torch.optim.Adam(Network.parameters(), lr=learning_rate)

    losses_path = os.path.join(output_training, 'losses.txt')
    best_dict_path = os.path.join(output_training, 'best_state_dict.pt')
    if os.path.isfile(losses_path) and not force:
        raise RuntimeError("Output file %s exists." % losses_path)
    with open(losses_path, 'w', buffering=1) as outfile:
        ### Training loop
        best_loss = 1.e10  # impossibly bad value
        for epoch in tqdm(range(1, epochs + 1), desc="Optimizing network", disable=not verbose, ascii=True):
            # Training epoch
            Network.train()
            training_running_loss = 0.
            training_batches = 0
            for training_samples, training_labels, _ in tqdm(TrainDL, desc="Iterating over training dataset",
                                                             leave=False, disable=not verbose, ascii=True):
                training_samples = training_samples.to(device=train_device)
                training_labels = training_labels.to(device=train_device)
                # Optimizer step on a single batch of training data
                opt.zero_grad()
                training_output = Network(training_samples)
                training_loss = loss(training_output, training_labels)
                training_loss.backward()
                # Clip gradients to make convergence somewhat easier
                torch.nn.utils.clip_grad_norm_(Network.parameters(), max_norm=clip_norm)
                # Make the actual optimizer step and save the batch loss
                opt.step()
                training_running_loss += training_loss.clone().cpu().item()
                training_batches += 1
            # Evaluation on the validation dataset
            Network.eval()
            with torch.no_grad():
                validation_running_loss = 0.
                validation_batches = 0
                for validation_samples, validation_labels, _ in tqdm(ValidDL, desc="Computing validation loss",
                                                                     leave=False, disable=not verbose, ascii=True):
                    validation_samples = validation_samples.to(device=train_device)
                    validation_labels = validation_labels.to(device=train_device)
                    # Evaluation of a single validation batch
                    validation_output = Network(validation_samples)
                    validation_loss = loss(validation_output, validation_labels)
                    validation_running_loss += validation_loss.clone().cpu().item()
                    validation_batches += 1
            # Print information on the training and validation loss in the current epoch and save current network state
            validation_loss = validation_running_loss / validation_batches
            output_string = '%04i    %f    %f' % (epoch, training_running_loss / training_batches, validation_loss)
            outfile.write(output_string + '\n')
            # Save 
            new_dict_path = os.path.join(output_training, 'state_dict_e_%04i.pt' % epoch)
            if os.path.isfile(new_dict_path) and not force:
                raise RuntimeError("Output file %s exists." % new_dict_path)
            torch.save(Network.state_dict(), new_dict_path)
            if validation_loss < best_loss:
                torch.save(Network.state_dict(), best_dict_path)
                best_loss = validation_loss

        logging.debug(("Training complete with best validation loss "
                       "%f, closing losses output file." % best_loss))
    return Network


def train_v2(Network, training_dataset, validation_dataset, output_training,
             num_params=3, batch_size=32, learning_rate=5e-5, epochs=100,
             clip_norm=100, verbose=False, force=False,
             train_device='cpu', num_workers=0):
    """Train a regression network to predict waveform parameters.

    Arguments
    ---------
    Network : RegressionNetwork (from get_base_network2)
        The network to train.
    training_dataset / validation_dataset : Dataset (or ConcatDataset)
        Must contain waveform_params; each __getitem__ returns
        (data, label, params) where label[:,0]==1 flags signal samples.
    output_training : str
        Directory for losses.txt and state dicts.
    num_params : {int, 3}
        How many of the 9 stored parameter columns to regress.
        Columns are taken in PARAM_COLUMNS order, so:
          3 → mass1, mass2, merger_time_in_window      (easiest)
          5 → + inclination, coa_phase
          7 → + pol_angle, declination
          9 → + right_ascension, merger_time_gps       (hardest)
    batch_size, learning_rate, epochs, clip_norm, verbose, force
        Same meaning as in train().
    train_device : {str, 'cpu'}
        Device to move each batch to before the forward pass.
    num_workers : {int, 0}
        Number of DataLoader worker processes for prefetching.
        Use 0 if store_device is CUDA.
    """
    logging.debug("Setting up datasets and data loaders for regression training.")

    # Log the active parameter targets
    active_cols = PARAM_COLUMNS[:num_params]
    logging.info("Regression targets (%d): %s", num_params, active_cols)

    pin = num_workers > 0 and 'cuda' in str(train_device)
    persistent = num_workers > 0
    TrainDL = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin,
                                          persistent_workers=persistent)
    ValidDL = torch.utils.data.DataLoader(validation_dataset, batch_size=500, shuffle=True,
                                          num_workers=num_workers, pin_memory=pin,
                                          persistent_workers=persistent)

    loss_fn = torch.nn.HuberLoss(reduction='mean')   # robust to outliers vs plain MSE
    opt = torch.optim.Adam(Network.parameters(), lr=learning_rate)

    losses_path = os.path.join(output_training, 'losses.txt')
    best_dict_path = os.path.join(output_training, 'best_state_dict.pt')
    if os.path.isfile(losses_path) and not force:
        raise RuntimeError("Output file %s exists." % losses_path)

    with open(losses_path, 'w', buffering=1) as outfile:
        # Write header
        outfile.write('# epoch    train_loss    val_loss    num_params=%d  cols=%s\n'
                      % (num_params, ','.join(active_cols)))

        best_loss = 1.e10
        for epoch in tqdm(range(1, epochs + 1), desc="Optimizing regression network",
                          disable=not verbose, ascii=True):
            # ── Training epoch ──────────────────────────────────────────────
            Network.train()
            tr_loss_sum, tr_batches = 0., 0
            for samples, labels, params in tqdm(TrainDL, desc="Training",
                                                leave=False, disable=not verbose, ascii=True):
                # Move entire batch to train_device in one shot (much faster than per-sample)
                samples = samples.to(device=train_device)
                labels  = labels.to(device=train_device)
                params  = params.to(device=train_device)

                # labels[:,0] == 1  →  signal sample (has valid params)
                signal_mask = labels[:, 0].bool()
                if signal_mask.sum() == 0:
                    continue  # skip all-noise batch (rare)

                opt.zero_grad()
                preds = Network(samples)                     # (B, num_params)
                # Only supervise on signal samples; params[:, :num_params] selects targets
                pred_sig  = preds[signal_mask]
                targ_sig  = params[signal_mask, :num_params].to(dtype=preds.dtype)
                batch_loss = loss_fn(pred_sig, targ_sig)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(Network.parameters(), max_norm=clip_norm)
                opt.step()
                tr_loss_sum += batch_loss.clone().cpu().item()
                tr_batches  += 1

            # ── Validation ──────────────────────────────────────────────────
            Network.eval()
            with torch.no_grad():
                val_loss_sum, val_batches = 0., 0
                for samples, labels, params in tqdm(ValidDL, desc="Validation",
                                                    leave=False, disable=not verbose, ascii=True):
                    samples = samples.to(device=train_device)
                    labels  = labels.to(device=train_device)
                    params  = params.to(device=train_device)

                    signal_mask = labels[:, 0].bool()
                    if signal_mask.sum() == 0:
                        continue
                    preds = Network(samples)
                    pred_sig = preds[signal_mask]
                    targ_sig = params[signal_mask, :num_params].to(dtype=preds.dtype)
                    val_loss_sum += loss_fn(pred_sig, targ_sig).clone().cpu().item()
                    val_batches  += 1

            tr_loss  = tr_loss_sum  / max(tr_batches,  1)
            val_loss = val_loss_sum / max(val_batches, 1)
            outfile.write('%04i    %f    %f\n' % (epoch, tr_loss, val_loss))

            new_dict_path = os.path.join(output_training, 'state_dict_e_%04i.pt' % epoch)
            if os.path.isfile(new_dict_path) and not force:
                raise RuntimeError("Output file %s exists." % new_dict_path)
            torch.save(Network.state_dict(), new_dict_path)
            if val_loss < best_loss:
                torch.save(Network.state_dict(), best_dict_path)
                best_loss = val_loss

        logging.debug("Regression training complete.  Best val loss: %f" % best_loss)
    return Network


def main():
    parser = ArgumentParser(
        description="CNN training script for submission to the MLGWSC-1 mock data challenge. Written by Ondřej Zelenka.")

    parser.add_argument('--verbose', action='store_true', help="Print update messages.")
    parser.add_argument('--debug', action='store_true', help="Show debug messages.")
    parser.add_argument('--force', action='store_true', help="Overwrite existing output file.")

    parser.add_argument('-d', '--dataset-file', type=str, nargs='+',
                        help="Path to the file where the datasets are stored.")
    parser.add_argument('-o', '--output-training', type=str,
                        help="Path to the directory where the outputs will be stored. The directory must exist.")
    parser.add_argument('-s', '--snr', type=float, nargs=2, default=(5., 15.),
                        help="Range from which the optimal SNRs will be drawn. Default: 5. 15.")
    parser.add_argument('-w', '--weights',
                        help="The path to the file containing the initial network weights. If empty, the weights are initialized randomly.")
    parser.add_argument('--coincident', action='store_true',
                        help="Train a set of networks for a coincident search. If omitted, a coherent search is trained.")
    parser.add_argument('--regression', action='store_true',
                        help="Train a regression network (get_base_network2) instead of the classification network.")
    parser.add_argument('--num-params', type=int, default=3,
                        help="Number of waveform parameter columns to regress (1-9, taken in PARAM_COLUMNS order). "
                             "Default: 3 (mass1, mass2, merger_time_in_window). Only used with --regression.")
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help="Learning rate of the optimizer. Default: 0.00001")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs. Default: 100")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size of the training algorithm. Default: 32")
    parser.add_argument('--clip-norm', type=float, default=100.,
                        help="Gradient clipping norm to stabilize the training. Default: 100.")
    parser.add_argument('--train-device', type=str, default='cpu',
                        help="Device to train the network. Use 'cuda' for the GPU. Also, 'cpu:0', 'cuda:1', etc. (zero-indexed). Default: cpu")
    parser.add_argument('--store-device', type=str, default='cpu',
                        help="Device to store the datasets. Use 'cuda' for the GPU. Also, 'cpu:0', 'cuda:1', etc. (zero-indexed). Default: cpu")
    parser.add_argument('--num-workers', type=int, default=4,
                        help="Number of DataLoader worker processes for prefetching batches. "
                             "Automatically set to 0 if --store-device is CUDA (CUDA tensors cannot be shared across processes). Default: 4")

    args = parser.parse_args()

    ### Set up logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s', level=log_level, datefmt='%d-%m-%Y %H:%M:%S')

    ### Load datasets
    TrainDS_list = []
    ValidDS_list = []
    for infname in args.dataset_file:
        logging.debug("Loading datasets from %s." % infname)
        TrainDS = Dataset(store_device=args.store_device, train_device=args.train_device, snr_range=args.snr)
        ValidDS = Dataset(store_device=args.store_device, train_device=args.train_device, snr_range=args.snr)
        with h5py.File(infname, 'r') as dataset_file:
            TrainDS.load(dataset_file, 'training')
            ValidDS.load(dataset_file, 'validation')
        TrainDS_list.append(TrainDS)
        ValidDS_list.append(ValidDS)
    TrainDS = torch.utils.data.ConcatDataset(TrainDS_list)
    ValidDS = torch.utils.data.ConcatDataset(ValidDS_list)
    logging.debug("Datasets loaded.")

    # CUDA tensors can't be shared across worker processes — force num_workers=0
    num_workers = 0 if 'cuda' in str(args.store_device) else args.num_workers
    if num_workers != args.num_workers:
        logging.info("store-device is CUDA: overriding --num-workers to 0")

    ### Initialize network
    logging.debug("Initializing network.")
    detectors = TrainDS[0][0].shape[0]
    if args.regression:
        num_params = min(max(args.num_params, 1), NUM_PARAMS)
        logging.info("Regression mode: predicting %d parameter(s): %s",
                     num_params, PARAM_COLUMNS[:num_params])
        Network = get_base_network2(path=args.weights, device=args.train_device,
                                    detectors=detectors, out_features=num_params)
    elif args.coincident:
        Network = get_coincident_network(path=args.weights, device=args.train_device, detectors=detectors)
    else:
        Network = get_coherent_network(path=args.weights, device=args.train_device, detectors=detectors)

    ### Train
    if args.regression:
        Network = train_v2(Network, TrainDS, ValidDS, args.output_training,
                           num_params=num_params,
                           batch_size=args.batch_size, learning_rate=args.learning_rate,
                           epochs=args.epochs, clip_norm=args.clip_norm,
                           verbose=args.verbose, force=args.force,
                           train_device=args.train_device, num_workers=num_workers)
    else:
        Network = train(Network, TrainDS, ValidDS, args.output_training,
                        batch_size=args.batch_size, learning_rate=args.learning_rate,
                        epochs=args.epochs, clip_norm=args.clip_norm,
                        verbose=args.verbose, force=args.force,
                        train_device=args.train_device, num_workers=num_workers)


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
