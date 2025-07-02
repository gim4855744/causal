from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

__all__ = ['SGCSL']


class SGCSL(nn.Module):

    """
    A Simple yet Scalable Granger Causal Structural Learning Approach for Topological Event Sequences.
    (https://proceedings.neurips.cc/paper_files/paper/2024/hash/b00efe6d4dbf119e2d0d44186cdfe7c8-Abstract-Conference.html)
    """

    def __init__(self, n_devices, n_events, k=2):

        """
        Parameters
        ----------
            n_devices: int
                Number of devices.
            n_events: int
                Number of events.
            k: int
                Connect k-hop nearest neighbors in device graph.
        """

        super().__init__()

        self._hparams = {
            'n_devices': n_devices,
            'n_events': n_events,
            'k': k
        }

        eps = 1e-7
        self._discovered_event_edges = nn.Parameter(self._revsoftplus(torch.full([n_events, n_events], eps)))
        self._mu = nn.Parameter(torch.full([n_events], eps))

    @staticmethod
    def _revsoftplus(x):
        return torch.log(torch.exp(x) - 1)
    
    @staticmethod
    def _find_k_hop_neighbors(adj, k):

        """
        Find k-hop neighbors for each node in the adjacency matrix.
        
        Parameters
        ----------
            adj: np.array
                Adjacency matrix of shape (n_nodes, n_nodes).
            k: int
                Number of hops.
            
        Returns
        -------
            neighbors: dict
                A dictionary where keys are node indices and values are lists of k-hop neighbors.
        """

        n_nodes = adj.shape[0]
        neighbors = {i: [i] for i in range(n_nodes)}
        
        for i in range(n_nodes):

            visited = [False] * n_nodes
            visited[i] = True
            queue = deque([(i, 0)])

            while queue:
                curr_node, curr_dist = queue.popleft()
                if curr_dist <= k:
                    for neighbor in range(n_nodes):
                        if adj[curr_node, neighbor] and not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append((neighbor, curr_dist + 1))
                            neighbors[i].append(neighbor)
        
        return neighbors
    
    def poisson_nll(self, A_mask1, mu_mask1, A_mask2, mu_mask2):
        A = F.softplus(self._discovered_event_edges)
        mu = F.softplus(self._mu)
        ll = torch.sum(A.unsqueeze(0) * A_mask1, dim=(1, 2)) + torch.sum(mu.unsqueeze(0) * mu_mask1, dim=1)
        ll = torch.sum(torch.log(ll))
        ll -= torch.sum(A * A_mask2)
        ll -= torch.sum(mu * mu_mask2)
        return -ll
    
    def _acyc_norm(self, n_events):
        A = F.softplus(self._discovered_event_edges)
        init_e = torch.eye(n_events)
        M = init_e + A * A / n_events
        E = torch.matrix_power(M, n_events - 1)
        h = (E.t() * M).sum() - n_events
        return h
    
    def _l1_norm(self):
        return torch.norm(F.softplus(self._discovered_event_edges), p=1)

    def forward(self):
        """
        Forward pass of the model.

        Returns
        -------
            discovered_event_edges: torch.Tensor
                Event edges.
            mu: torch.Tensor
                Poisson strength of each event.
        """
        return self._discovered_event_edges, self._mu

    def fit(self, x, device_edges, lr=5e-4, max_epochs=100, ckptpath='./checkpoints/SGCSL.ckpt'):

        """
        Fit the model to the data.

        Parameters
        ----------
            x: pd.DataFrame
                Observed events.
            device_edges: np.ndarray
                Device adjacency matrix.
            lr: float
                Learning rate for the optimizer.
            max_epochs: int
                Number of epochs to train.
            num_workers: int
                How many subprocesses to use for data.
        """
        
        self.train()

        n_events = self._hparams['n_events']

        cols = ['event', 'start_timestamp', 'end_timestamp']
        device = next(iter(self.parameters())).device

        neighbors = self._find_k_hop_neighbors(device_edges, self._hparams['k'])
        seqs, neighbor_seqs = [], []

        # Prepare data
        for device in range(self._hparams['n_devices']):
            filtered_x = x[x['device'] == device]
            seqs.append(filtered_x[cols].values)
            neighbor_filtered_x = x[x['device'].isin(neighbors[device])]
            neighbor_seqs.append(neighbor_filtered_x[cols].values)

        # Get mask
        mask_len, Tc = 0, 0
        for s in seqs:
            mask_len += s.shape[0]
            Tc += s[-1, 2]  # ?
        A_mask1 = np.zeros((mask_len, n_events, n_events))
        mu_mask1 = np.zeros((mask_len, n_events))
        A_mask2 = np.zeros((n_events, n_events))
        mu_mask2 = np.ones(n_events) * Tc

        def count_bin(g):
            counts = np.bincount(g)
            if counts.shape[0] < n_events:
                counts = np.concatenate([counts, np.zeros(n_events - counts.shape[0])])
            return counts

        count = 0
        for seq, neighbor_seq in zip(seqs, neighbor_seqs):
            for e in range(seq.shape[0]):
                time_e = seq[e, 1]
                g = neighbor_seq[(neighbor_seq[:, 1] < time_e) & (neighbor_seq[:, 2] > time_e), 0]
                A_mask1[count, :, seq[e, 0]] = count_bin(g)
                mu_mask1[count, seq[e, 0]] = 1
                count += 1
            for neighbor_e in range(neighbor_seq.shape[0]):
                A_mask2[neighbor_seq[neighbor_e, 0], :] += neighbor_seq[neighbor_e, 2] - neighbor_seq[neighbor_e, 1]

        device = next(iter(self.parameters())).device

        A_mask1 = torch.tensor(A_mask1, dtype=torch.float32, device=device)
        mu_mask1 = torch.tensor(mu_mask1, dtype=torch.float32, device=device)
        A_mask2 = torch.tensor(A_mask2, dtype=torch.float32, device=device)
        mu_mask2 = torch.tensor(mu_mask2, dtype=torch.float32, device=device)

        optimizer = torch.optim.Adam(self.parameters(), lr)

        tqdm_epoch = tqdm(range(1, max_epochs + 1))
        for epoch in tqdm_epoch:
            tqdm_epoch.set_description_str(f'Epoch {epoch}')
            loss = self.poisson_nll(A_mask1, mu_mask1, A_mask2, mu_mask2) + 5e3 * self._l1_norm() + 5e3 * self._acyc_norm(n_events)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tqdm_epoch.set_postfix_str(f'Loss: {loss.item():.4f}')
            checkpoint = {
                'model_state_dict': self.state_dict(),
                'hparams': self._hparams
            }
            torch.save(checkpoint, ckptpath)

    def predict(self):
        """
        Predict the event edges and Poisson strength of each event.

        Returns
        -------
            discovered_event_edges: torch.Tensor
                Event edges.
            mu: torch.Tensor
                Poisson strength of each event.
        """
        self.eval()
        return F.softplus(self._discovered_event_edges).detach().cpu().numpy(), self._mu.detach().cpu().numpy()
