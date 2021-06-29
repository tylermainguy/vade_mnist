"""Module for decoder for VAE."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Decoder(nn.Module):
    """Decoder module for VAE."""

    def __init__(self, args: dict):
        """
        Parameters
        ----------
        args : dict
            Mapping containing parameters for network initialization
        ----------

        """

        super().__init__()

        # network architecture
        self.decoder_input = nn.Linear(args.latent, args.h2_size)
        self.hidden_1 = nn.Linear(args.h2_size, args.h1_size)
        self.hidden_2 = nn.Linear(args.h1_size, args.h1_size)
        self.decoder_output = nn.Linear(args.h1_size, args.input_size)

    def forward(self, latent: Tensor):
        """Forward pass of decoder network.

        Parameters
        ----------
        latent : torch.Tensor
            Torch Tensor containing the sampled latent vector.
        ----------

        Returns
        ----------
        Tensor
            Reconstructed input from latent vector.
        ----------

        """

        data = F.relu(self.decoder_input(latent))
        data = F.relu(self.hidden_1(data))
        data = F.relu(self.hidden_2(data))

        reconstructed = torch.sigmoid(self.decoder_output(data))

        return reconstructed
