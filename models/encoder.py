"""Module for encoder model of VAE."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Encoder(nn.Module):
    """
    Encoder module for VAE.
    """

    def __init__(self, args: dict):
        """
        Parameters
        ----------
        params : dict
            Mapping containing parameters for network initialization
        ----------
        """
        super().__init__()

        # network architecture
        self.encoder_input = nn.Linear(args.input_size, args.h1_size)
        self.hidden_1 = nn.Linear(args.h1_size, args.h1_size)
        self.hidden_2 = nn.Linear(args.h1_size, args.h2_size)

        # latent variable means and log variances
        self.means = nn.Linear(args.h2_size, args.latent)
        self.log_vars = nn.Linear(args.h2_size, args.latent)

    def forward(self, data: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass of encoder network.

        Parameters
        ----------
        data : Tensor
            Samples to be encoded.
        ----------

        Returns
        ----------
        (Tensor, Tensor, Tensor)
            Tuple of latent vector, latent variable distribution means and variances.
        ----------

        """
        # encoding
        data = F.relu(self.encoder_input(data))
        data = F.relu(self.hidden_1(data))
        data = F.relu(self.hidden_2(data))

        means = self.means(data)
        log_vars = self.log_vars(data)

        latent = self.reparameterize(means, log_vars)

        return latent, means, log_vars

    def reparameterize(self, means: Tensor, log_vars: Tensor):
        """
        Use the reparameterization trick to randomly sample from the latent
        distributions while still allowing for backpropogation.

        Parameters
        ----------
        means : Tensor
            Data containing latent variable distribution means.
        log_vars : Tensor
            Vector containing latent variable distribution log variances.
        ----------

        Returns
        ----------
        Tensor
            Data sampled from distributions defined by means and log_vars.
        ----------

        """
        # converting from log variance to variance
        sampled_vars = torch.exp(0.5 * log_vars)

        # reparameterization trick
        eps = torch.randn_like(log_vars)

        return means + sampled_vars * eps
