import numpy as np
import torch
import torch.nn as nn


class RNNACE(nn.Module):

    """
    Neural Network Attributions: A Causal Perspective
    (https://proceedings.mlr.press/v97/chattopadhyay19a)
    """

    def __init__(self, predictor, task):

        """
        Parameters
        ----------
        predictor: nn.Module
            PyTorch model that wants to estimate causal effects.
        task: str
            One of 'regression', 'binary' or 'classification'.
        """

        super().__init__()

        self._hparams = {
            'predictor': predictor,
            'task': task
        }

    def forward(self, x, t, feature_idx, output_idx, alpha):

        """
        Parameters
        ----------
        x: np.ndarray
            Source(training) data of shape (n_samples, n_feats)
        t: int
            Index of the time step to intervene on.
        feature_idx: int
            Index of the feature to intervene on.
        output_idx: int
            Index of the class to intervene on.
        alpha: float
            Intervention value (do(alpha)).

        Returns
        -------
        causal_effect: float
            Estimated causal effect.
        """

        model = self._hparams['predictor']
        model.eval()
        model.zero_grad()
        n_features = x.shape[2]

        intervened_x = x[:, :t + 1].copy()
        intervened_x[:, t, feature_idx] = alpha

        intervened_means = np.mean(intervened_x, axis=0, keepdims=True)
        intervened_means = torch.tensor(intervened_means, dtype=torch.float32, requires_grad=True)
        intervened_covs = np.cov(intervened_x.reshape(len(intervened_x), -1), rowvar=False)
        prediction = model(intervened_means)
        if self._hparams['task'] == 'binary':
            prediction = torch.sigmoid(prediction)
        elif self._hparams['task'] == 'classification':
            prediction = torch.softmax(prediction, dim=-1)

        grad_mask = torch.zeros_like(prediction)
        grad_mask[output_idx] = 1.0
        first_grads = torch.autograd.grad(prediction, intervened_means, grad_mask, retain_graph=True, create_graph=True)
        expectation_on_alpha = prediction[output_idx].item()

        for i in range(t + 1):
            for j in range(n_features):
                grad_mask = torch.zeros_like(first_grads[0])
                grad_mask[:, i, j] = 1.0
                hessian = torch.autograd.grad(first_grads, intervened_means, grad_mask, retain_graph=True, create_graph=True)[0]
                hessian = hessian.detach().cpu().numpy()
                expectation_on_alpha += 0.5 * np.sum(hessian * intervened_covs[i * n_features + j].reshape(hessian.shape))

        return expectation_on_alpha

    def fit(self):
        return
    
    def predict(self, x, t, feature_idx, output_idx, alpha):
        
        """
        Predict causal effect.
        
        Parameters
        ----------
        x: np.ndarray
            Source(training) data of shape (n_samples, n_feats)
        t: int
            Index of the time step to intervene on.
        feature_idx: int
            Index of the feature to intervene on.
        output_idx: int
            Index of the class to intervene on.
        alpha: float
            Intervention value (do(alpha)).

        Returns
        -------
        causal_effect: float
            Estimated causal effect.
        """
        
        return self(x, t, feature_idx, output_idx, alpha)
