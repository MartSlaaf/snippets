import torch
from torch import nn
import numpy as np


class NoisyEmbedding(nn.Embedding):
    """
    Embeddings with additive gaussian noise with mean=0 and user-defined variance.
    *args and **kwargs defined by usual Embeddings
    Args:
        noise_scale (float): when > 0 applies additive noise to embeddings. When = 0, forward is equivalent to usual embeddings.
        dropout (float): probability of embedding axis to be dropped. 0 means no dropout at all.

    For other parameters defenition look at nn.Embedding help
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                 norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None,
                 noise_scale=0, dropout=0):
        super().__init__(num_embeddings, embedding_dim, padding_idx=None, max_norm=None,
                         norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None)
        self.noise = torch.distributions.Normal(0, noise_scale)
        self.scale = noise_scale
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = super().forward(x)
        if self.training and self.scale > 0:
            x += self.noise.sample((self.weight.shape[1], )).to(self.weight.device)
        x = self.dropout(x)
        return x
