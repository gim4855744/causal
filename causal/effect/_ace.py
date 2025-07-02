import numpy as np
import torch
import torch.nn as nn


class ACE(nn.Module):

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
            one of 'regression', 'binary' or 'classification'.
        """

        super().__init__()

        self._hparams = {
            'predictor': predictor,
            'task': task
        }
    
    def forward(self, x, feature_idx, output_idx, alpha):

        """
        Parameters
        ----------
        x: np.ndarray
            Source(training) data of shape (n_samples, n_features)
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

        predictor = self._hparams['predictor']
        predictor.eval()
        predictor.zero_grad()
        n_features = x.shape[1]

        cov = np.cov(x, rowvar=False)
        means = np.mean(x, axis=0)

        intervened_means = means.copy()
        intervened_means[feature_idx] = alpha
        intervened_means = torch.tensor(intervened_means, dtype=torch.float32, requires_grad=True)
        
        prediction = predictor(intervened_means)
        if self._hparams['task'] == 'binary':
            prediction = torch.sigmoid(prediction)
        elif self._hparams['task'] == 'classification':
            prediction = torch.softmax(prediction, dim=-1)
        expectation_on_alpha = prediction[output_idx].item()  # 1st term in the interventional expectation
        
        grad_mask = torch.zeros_like(prediction)
        grad_mask[output_idx] = 1
        first_grad = torch.autograd.grad(prediction, intervened_means, grad_mask, retain_graph=True, create_graph=True)

        for i in range(n_features):  # Tr(Hessian * Cov)
            
            if i != feature_idx:

                intervened_cov = cov.copy()
                intervened_cov[i, feature_idx] = 0

                grad_mask = torch.zeros(n_features, dtype=torch.float32)
                grad_mask[i] = 1

                hessian = torch.autograd.grad(first_grad, intervened_means, grad_mask, retain_graph=True, create_graph=True)[0]
                hessian = hessian.detach().numpy()

                expectation_on_alpha += np.sum(0.5 * hessian * intervened_cov[i])  # adding the 2nd term in the interventional expectation

        return expectation_on_alpha

    def fit(self):
        return
    
    def predict(self, x, feature_idx, output_idx, alpha):
        
        """
        Predict causal effect.

        Parameters
        ----------
        x: np.ndarray
            Source(training) data of shape (n_samples, n_features)
        feature_idx: int
            Index of the feature want to intervene in.
        output_idx: int
            Index of the class want to intervene in.
        alpha: float
            Intervention value (do(alpha)).
        
        Returns
        -------
        causal_effect: float
            Estimated causal effect.
        """
        
        return self(x, feature_idx, output_idx, alpha)
