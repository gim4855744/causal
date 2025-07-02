import bisect
from random import shuffle

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

__all__ = ['ACDSS']


class _TimeseriesDataset(Dataset):

    def __init__(self, x: pd.DataFrame):
        super().__init__()
        self._x = [torch.tensor(v.values, dtype=torch.float32).unsqueeze(dim=-1) for _, v in x.groupby(x.index.name)]

    def __getitem__(self, index):
        return self._x[index]
    
    def __len__(self):
        return len(self._x)


class _BucketSampler(Sampler):

    def __init__(self, data_source, batch_size):

        self._data_source = data_source
        self._batch_size = batch_size

        data_lengths = [len(xi) for xi in data_source]  # lenghts of samples
        unique_data_lengths = sorted(set(data_lengths))  # set of sample lengths
        length_frequencies = {length: 0 for length in unique_data_lengths}
        for length in data_lengths:  # get sorted length frequencies
            length_frequencies[length] += 1

        n = 0
        bucket_boundaries = []
        for length, frequency in length_frequencies.items():  # get bucket boundaries automatically
            n += frequency
            if n >= batch_size:
                bucket_boundaries.append(length)
                n = 0
                
        self._buckets = {bucket_id: [] for bucket_id in range(len(bucket_boundaries) + 1)}
        for i, length in enumerate(data_lengths):  # bucketing data
            bucket_id = bisect.bisect_left(bucket_boundaries, length)
            self._buckets[bucket_id].append(i)

        self._iter_list = []

        for bucket_id in self._buckets.keys():

            bucket = self._buckets[bucket_id]

            if len(bucket) > 0:

                n_splits = len(bucket) // self._batch_size
                if len(bucket) % self._batch_size > 0:
                    n_splits += 1

                split_bucket = np.array_split(bucket, n_splits)
                self._iter_list.extend(split_bucket)

    def __iter__(self):

        for bucket_id in self._buckets.keys():
            shuffle(self._buckets[bucket_id])
        shuffle(self._iter_list)

        for batch in self._iter_list:
            yield batch
    
    def __len__(self):
        return len(self._iter_list)


def _pad_sequence_collate(batch):
    n_steps = torch.tensor([len(xi) for xi in batch], dtype=torch.int64)
    x = pad_sequence(batch, batch_first=True)
    return x, n_steps


def _get_dataloder(x, batch_size: int = 1, num_workers: int = 0):
    data_source = [v.values for _, v in x.groupby(x.index.name)]
    dataset = _TimeseriesDataset(x)
    sampler = _BucketSampler(data_source, batch_size)
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=_pad_sequence_collate, num_workers=num_workers)
    return dataloader


class _MLP(nn.Module):

    """
    Three-Layer Perceptron with ELU non-linearity and batch normalization.
    """

    def __init__(self, in_dim, out_dim, n_hidden, batchnorm=True):

        """
        Parameters
        ----------
            in_dim: int
                Size of input dimension.
            out_dim: int
                Size of output dimension.
            n_hidden: int
                Number of hidden units.
            batchnorm: bool
                Whether to use batch normalization.
        """

        super().__init__()

        self._linear1 = nn.Linear(in_dim, n_hidden)
        self._linear2 = nn.Linear(n_hidden, n_hidden)
        self._linear3 = nn.Linear(n_hidden, out_dim)

        self._batchnorm = None
        if batchnorm:
            self._batchnorm = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        
        x = F.elu(self._linear1(x))
        x = F.elu(self._linear2(x))
        x = self._linear3(x)

        if self._batchnorm is not None:
            shape = x.shape
            x = x.view(-1, shape[-1])
            x = self._batchnorm(x)
            x = x.view(shape)

        return x


class _RNN(nn.Module):

    """GRU-style N-dimentional RNN"""

    def __init__(self, in_size, hidden_size):

        super().__init__()
        
        self._input_projection = nn.Linear(in_size, hidden_size)
        self._gate1 = nn.Linear(2 * hidden_size, hidden_size)
        self._gate2 = nn.Linear(2 * hidden_size, hidden_size)
        self._output_projection = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, x, prev_hidden):

        # x: (batch_size, n_steps, n_series, 1)
        # hidden: (batch_size, n_steps, n_series, hidden_size)
        
        x = self._input_projection(x)
        gate_x = torch.concat([x, prev_hidden], dim=-1)
        gate1 = torch.sigmoid(self._gate1(gate_x))
        gate2 = torch.sigmoid(self._gate2(gate_x))

        h = torch.concat([x, gate1 * prev_hidden], dim=-1)
        h = torch.tanh(self._output_projection(h))
        h = (1 - gate2) * prev_hidden + gate2 * h

        return h
    

class _Encoder(nn.Module):

    """Encoder of ACDSS."""

    def __init__(self, n_feats, n_hidden, n_edge_types):

        """
        Parameters
        ----------
            n_steps: int
                Number of time steps of a time series.
            n_feats: int
                Number of features in a time series.
            n_hidden: int
                Number of hidden units.
            n_edge_types: int
                Number of edge types (a.k.a. event types).
        """

        super().__init__()

        self._mlp1 = _MLP(n_feats, n_hidden, n_hidden)
        self._mlp2 = _MLP(2 * n_hidden, n_hidden, n_hidden)
        self._mlp3 = _MLP(n_hidden, n_hidden, n_hidden)
        self._mlp4 = _MLP(3 * n_hidden, n_hidden, n_hidden)
        self._out_layer = nn.Linear(n_hidden, n_edge_types)

    def _edge2node(self, x, receiver):
        x = receiver.T @ x
        return x / x.size(1)
    
    def _node2edge(self, x, receiver, sender):
        receiver = receiver @ x
        sender = sender @ x
        return torch.concat([sender, receiver], dim=2)

    def forward(self, x, receiver, sender):
        
        # input shape: [batch_size, n_series, n_feats]

        x = self._mlp1(x)
        x = self._node2edge(x, receiver, sender)
        x = self._mlp2(x)
        x_skip = x

        x = self._edge2node(x, receiver)
        x = self._mlp3(x)
        x = self._node2edge(x, receiver, sender)
        x = torch.concat([x, x_skip], dim=2)
        x = self._mlp4(x)

        return self._out_layer(x)
    

class _Decoder(nn.Module):

    """Decoder of ACDSS."""

    def __init__(self, n_feats, n_hidden, n_edge_types):

        """
        Parameters
        ----------
            n_feats: int
                Number of features in a time series.
            n_hidden: int
                Number of hidden units.
            n_edge_types: int
                Number of edge types (a.k.a. event types).
        """

        super().__init__()

        self._n_hidden = n_hidden
        self._msg_net = nn.ModuleList([_MLP(2 * n_feats, n_hidden, n_hidden, batchnorm=False) for _ in range(n_edge_types)])
        self._out_net = _MLP(n_feats + n_hidden, n_feats, n_hidden, batchnorm=False)
        
    def _node2edge(self, x, receiver, sender):
        receiver = receiver @ x
        sender = sender @ x
        return torch.concat([sender, receiver], dim=3)

    def forward(self, x, edges, receiver, sender):
        
        batch_size, n_edges, n_edge_types = edges.shape
        n_steps = x.size(1)
        
        edges = edges.unsqueeze(dim=1).expand([batch_size, n_steps, n_edges, n_edge_types])

        all_msg = torch.zeros([batch_size, n_steps, n_edges, self._n_hidden], device=x.device)
        init_msg = self._node2edge(x, receiver, sender)
        for i in range(1, n_edge_types):  # skip first edge type
            msg = self._msg_net[i](init_msg)
            msg = msg * edges[:, :, :, i: i + 1]
            all_msg += msg

        agg_msg = all_msg.transpose(2, 3) @ receiver
        agg_msg = agg_msg.transpose(2, 3).contiguous()

        aug_x = torch.concat([x, agg_msg], dim=-1)
        pred = x + self._out_net(aug_x)

        return pred


class ACDSS(nn.Module):

    """
    ACDSS (ACD for CDSS) is a simple tweak of ACD to handle dynamic time series.
    It first encodes the time series into a global representation using a recurrent neural network (RNN).
    """

    def __init__(self, n_series, n_hidden, n_edge_types):

        """
        Parameters
        ----------
            n_series: int
                Number of time series.
            n_hidden: int
                Number of hidden units.
            n_edge_types: int
                Number of edge types (a.k.a. event types).
        """

        super().__init__()

        self._hparams = {
            'n_series': n_series,
            'n_hidden': n_hidden,
            'n_edge_types': n_edge_types
        }

        # create receiver-sender relations of the fully-connected graph (w/o self-connection)
        diag = torch.eye(n_series, dtype=torch.int64)
        off_diag = torch.ones([n_series, n_series], dtype=torch.int64) - diag
        receiver, sender = torch.where(off_diag)
        self._receiver = diag[receiver].clone().detach().to(torch.float32)
        self._sender = diag[sender].clone().detach().to(torch.float32)

        # create encoder/decoder
        self._rnn = _RNN(1, n_hidden)
        self._encoder = _Encoder(n_hidden, n_hidden, n_edge_types)
        self._decoder = _Decoder(1, n_hidden, n_edge_types)

    def _kld_uniform(self, edge_probs, eps=1e-16):
        """Compute the Kullback-Leibler Divergence between the edge probabilities and a uniform distribution."""
        v = edge_probs * torch.log(edge_probs + eps)
        return v.sum() / (edge_probs.size(0) * self._hparams['n_series'])
    
    def _gaussian_nll(self, preds, target, mask, forecasting_length=1, variance=5e-7):

        """Compute the negative log-likelihood of a Gaussian distribution."""

        batch_size = preds.size(0)

        preds = preds[:, :-forecasting_length]
        target = target[:, forecasting_length:]
        mask = (mask[:, :-forecasting_length] & mask[:, forecasting_length:])  # check both preds and target are exist
        
        preds = preds.masked_select(mask)
        target = target.masked_select(mask)

        neg_log_p = (preds - target) ** 2 / (2 * variance)

        return neg_log_p.sum() / (batch_size * self._hparams['n_series'])

    def forward(self, x, n_steps):

        """
        Forward pass of the model.

        Parameters
        ----------
            x: torch.Tensor
                Input tensor of shape (batch_size, n_series, n_steps, n_feats).
            n_steps: torch.Tensor
                Tensor of shape (batch_size,) containing the number of time steps for each sample.

        Returns
        -------
            edge_probs: torch.Tensor
                Predicted edge probabilities of shape (batch_size, n_series, n_series, n_edge_types).
            pred: torch.Tensor
                Reconstructed time series of shape (batch_size, n_series, n_steps, n_feats).
        """

        max_steps = x.size(1)
        hidden = torch.zeros(x.size(0), x.size(2), self._hparams['n_hidden'], device=x.device)
        hidden_states = []
        for t in range(max_steps):
            hidden_states.append(self._rnn(x[:, t], hidden))
        hidden_states = torch.stack(hidden_states, dim=1)
        hidden_states = hidden_states[torch.arange(x.size(0), device=x.device), n_steps - 1]  # zero index

        receiver = self._receiver.to(x.device)
        sender = self._sender.to(x.device)

        logits = self._encoder(hidden_states, receiver, sender)  # encode 'x' to edge logits

        predicted_edges = F.gumbel_softmax(logits, tau=0.5, dim=-1)  # discovered edges for training
        edge_probs = F.softmax(logits, dim=-1)  # discovered edges

        pred = self._decoder(x, predicted_edges, receiver, sender)  # decode 'edges' to 'x'

        return edge_probs, pred
    
    def fit(self, x, batch_size=1, lr=5e-4, max_epochs=100, num_workers=0, ckptpath='./checkpoints/ACDSS.ckpt'):

        """
        Fit the model to the data.

        Parameters
        ----------
            x: pd.DataFrame
                DataFrame of shape (n_samples * n_steps, n_series). Each subject is identified by a unique ID and ordered chronologically.
            batch_size: int
                Batch size for training.
            lr: float
                Learning rate for the optimizer.
            max_epochs: int
                Number of epochs to train.
            num_workers: int
                How many subprocesses to use for data.
            ckptpath: str
                Path to save the model checkpoint.
        """

        self.train()

        device = next(iter(self.parameters())).device
        
        # Prepare dataloader.
        dataloader = _get_dataloder(x, batch_size, num_workers)

        # Configure optimizer.
        optimizer = torch.optim.Adam(self.parameters(), lr)

        # Training loop.

        tqdm_epoch = tqdm(range(1, max_epochs + 1))
        loss_history = []

        for epoch in tqdm_epoch:

            tqdm_epoch.set_description_str(f'Epoch {epoch}')
            tqdm_batch = tqdm(dataloader, leave=False)
            losses = []

            for batch_x, batch_n_steps in tqdm_batch:

                batch_x = batch_x.to(device)
                batch_n_steps = batch_n_steps.to(device)

                edge_probs, pred = self(batch_x, batch_n_steps)

                step_arange = torch.arange(batch_x.size(1), device=batch_x.device).expand(batch_x.size(0), batch_x.size(3), batch_x.size(2), batch_x.size(1)).transpose(1, 3)
                step_mask = step_arange <= batch_n_steps.view(-1, 1, 1, 1)

                loss_kld = self._kld_uniform(edge_probs)
                loss_nll = self._gaussian_nll(pred, batch_x, step_mask)
                loss = loss_kld + loss_nll

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tqdm_batch.set_postfix_str(f'Loss: {loss.item():.4f}')
                losses.append(loss.item())

            loss_history.append(np.mean(losses))
            checkpoint = {
                'loss': loss_history,
                'model_state_dict': self.state_dict(),
                'hparams': self._hparams
            }
            torch.save(checkpoint, ckptpath)
    
    def predict(self, x, batch_size=1, num_workers=0):

        """
        Predict the edge probabilities.

        Parameters
        ----------
            x: pd.DataFrame
                pandas DataFrame of shape (n_samples * n_steps, n_series).
            batch_size: int
                Batch size for prediction.
            num_workers: int
                How many subprocesses to use for data.

        Returns
        -------
            transformed_edge_probs: np.ndarray
                Predicted edge probabilities of shape (n_samples, n_series, n_series, n_edge_types).
        """

        self.eval()

        device = next(iter(self.parameters())).device

        # Prepare data.
        dataloader = _get_dataloder(x, batch_size, num_workers)

        # Prediction.
        total_edge_probs = []
        for batch_x, batch_n_steps in tqdm(dataloader, leave=False):
            batch_x = batch_x.to(device)
            batch_n_steps = batch_n_steps.to(device)
            edge_probs, _ = self(batch_x, batch_n_steps)
            total_edge_probs.append(edge_probs.detach().cpu())

        # Transform edge probatilities to the original shape.
        n_samples, n_series = x.index.nunique(), x.shape[1]
        total_edge_probs = torch.concat(total_edge_probs, dim=0)
        total_edge_probs = total_edge_probs[:, :, 1].reshape(-1, n_series, n_series - 1)
        transformed_edge_probs = torch.zeros([n_samples, n_series, n_series], dtype=torch.float32)
        mask = ~torch.eye(n_series, dtype=torch.bool).unsqueeze(dim=0).expand(n_samples, -1, -1)
        transformed_edge_probs[mask] = total_edge_probs.ravel()

        return transformed_edge_probs.detach().cpu().numpy()
