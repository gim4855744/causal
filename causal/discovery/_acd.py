import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

__all__ = ['ACD']


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
    

class _Encoder(nn.Module):

    """Encoder of ACD."""

    def __init__(self, n_steps, n_feats, n_hidden, n_edge_types):

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

        self._mlp1 = _MLP(n_steps * n_feats, n_hidden, n_hidden)
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
        
        # input shape: [batch_size, n_series, n_steps, n_feats]
        # new shape: [batch_size, n_series, n_steps * n_feats]
        batch_size, n_series = x.size(0), x.size(1)
        x = x.reshape(batch_size, n_series, -1)

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

    """Decoder of ACD."""

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
        n_steps = x.size(2)

        x = x.transpose(1, 2).contiguous()
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

        return pred.transpose(1, 2).contiguous()


class ACD(nn.Module):

    """
    Amortized Causal Discovery: Learning to Infer Causal Graphs from Time-Series Data.
    (https://proceedings.mlr.press/v177/lowe22a.html)
    """

    def __init__(self, n_series, n_steps, n_feats, n_hidden, n_edge_types):

        """
        Parameters
        ----------
            n_series: int
                Number of time series.
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

        self._hparams = {
            'n_series': n_series,
            'n_steps': n_steps,
            'n_feats': n_feats,
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
        self._encoder = _Encoder(n_steps, n_feats, n_hidden, n_edge_types)
        self._decoder = _Decoder(n_feats, n_hidden, n_edge_types)

    def _kld_uniform(self, edge_probs, eps=1e-16):
        """Compute the Kullback-Leibler Divergence between the edge probabilities and a uniform distribution."""
        v = edge_probs * torch.log(edge_probs + eps)
        return v.sum() / (edge_probs.size(0) * self._hparams['n_series'])
    
    def _gaussian_nll(self, preds, target, variance=5e-7):
        """Compute the negative log-likelihood of a Gaussian distribution."""
        neg_log_p = (preds - target) ** 2 / (2 * variance)
        return neg_log_p.sum() / (target.size(0) * self._hparams['n_series'])  # target.size(0) is batch size.

    def forward(self, x):

        """
        Forward pass of the model.

        Parameters
        ----------
            x: torch.Tensor
                Input tensor of shape (batch_size, n_series, n_steps, n_feats).

        Returns
        -------
            edge_probs: torch.Tensor
                Predicted edge probabilities of shape (batch_size, n_series, n_series, n_edge_types).
            pred: torch.Tensor
                Reconstructed time series of shape (batch_size, n_series, n_steps, n_feats).
        """

        receiver = self._receiver.to(x.device)
        sender = self._sender.to(x.device)

        logits = self._encoder(x, receiver, sender)  # encode 'x' to edge logits

        predicted_edges = F.gumbel_softmax(logits, tau=0.5, dim=-1)  # discovered edges for training
        edge_probs = F.softmax(logits, dim=-1)  # discovered edges

        pred = self._decoder(x, predicted_edges, receiver, sender)  # decode 'edges' to 'x'

        return edge_probs, pred
    
    def fit(self, x, batch_size=1, lr=5e-4, max_epochs=100, num_workers=0, ckptpath='./checkpoints/ACD.ckpt'):

        """
        Fit the model to the data.

        Parameters
        ----------
            x: np.ndarray
                Input np.ndarray of shape (n_samples, n_series, n_steps, n_feats).
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
        x = torch.tensor(x, dtype=torch.float32, device=device)
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers)

        # Configure optimizer.
        optimizer = torch.optim.Adam(self.parameters(), lr)

        # Training loop.
        tqdm_epoch = tqdm(range(1, max_epochs + 1))
        for epoch in tqdm_epoch:
            tqdm_epoch.set_description_str(f'Epoch {epoch}')
            tqdm_batch = tqdm(dataloader, leave=False)
            for batch in tqdm_batch:
                batch_x = batch[0].to(device)
                edge_probs, pred = self(batch_x)
                loss_kld = self._kld_uniform(edge_probs)
                loss_nll = self._gaussian_nll(pred[:, :, :-1], batch_x[:, :, 1:])  # 1-step ahead forecasting.
                loss = loss_kld + loss_nll
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tqdm_batch.set_postfix_str(f'Loss: {loss.item():.4f}')
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'hparams': self._hparams
            }
            torch.save(checkpoint, ckptpath)
    
    def predict(self, x, batch_size=1, num_workers=0):

        """
        Predict the edge probabilities.

        Parameters
        ----------
            x: np.ndarray
                Input np.ndarray of shape (n_samples, n_series, n_steps, n_feats).
            batch_size: int
                Batch size for training.
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
        x = torch.tensor(x, dtype=torch.float32, device=device)
        dataset = TensorDataset(x)
        dataloader = DataLoader(dataset, batch_size, num_workers=num_workers)

        # Prediction.
        total_edge_probs = []
        for batch in tqdm(dataloader, leave=False):
            batch_x = batch[0].to(device)
            edge_probs, _ = self(batch_x)
            total_edge_probs.append(edge_probs.detach().cpu())

        # Transform edge probatilities to the original shape.
        n_samples, n_series = x.size(0), x.size(1)
        total_edge_probs = torch.concat(total_edge_probs, dim=0)
        total_edge_probs = total_edge_probs[:, :, 1].reshape(-1, n_series, n_series - 1)
        transformed_edge_probs = torch.zeros([n_samples, n_series, n_series], dtype=torch.float32)
        mask = ~torch.eye(n_series, dtype=torch.bool).unsqueeze(dim=0).expand(n_samples, -1, -1)
        transformed_edge_probs[mask] = total_edge_probs.ravel()

        return transformed_edge_probs.detach().cpu().numpy()
