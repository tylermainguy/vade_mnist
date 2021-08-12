"""Module for Vade."""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from utils import reparameterize

from models.decoder import Decoder
from models.encoder import Encoder


class VaDE(nn.Module):
    """VaDE clustering model."""

    def __init__(self, args: dict, encoder=None, decoder=None):
        """
        Parameters
        ----------
        args : dict
            Mapping containing parameters for network initialization.
        ----------

        """
        super().__init__()

        # initialiaze symmetric encoder-decoder
        if not encoder:
            self.encoder = Encoder(args)
        else:
            self.encoder = encoder
        if not decoder:
            self.decoder = Decoder(args)
        else:
            self.decoder = decoder

        # intialize clusters for GMM
        self.pi_prior = nn.Parameter(
            torch.FloatTensor(args.n_clusters).fill_(1) / args.n_clusters,
            requires_grad=True,
        )
        self.mu_prior = nn.Parameter(
            torch.FloatTensor(args.n_clusters, args.latent).fill_(0),
            requires_grad=True,
        )
        self.log_var_prior = nn.Parameter(
            torch.FloatTensor(args.n_clusters, args.latent).fill_(0),
            requires_grad=True,
        )

    def forward(self, data: Tensor):
        """
        Forward pass through VAE + clustering.

        Paramaters
        ----------
        data : torch.Tensor
            Torch tensor containing training samples.
        ----------

        Returns
        ----------
        (Tensor, Tensor, Tensor, Tensor)
            reconstructed input, latent vector, mean and log variances
        ----------

        """

        # encoder
        means, log_vars = self.encoder(data)

        latent = reparameterize(means, log_vars)

        # decoder
        x_hat = self.decoder(latent)

        return x_hat, latent, means, log_vars

    def cluster_probabilities(self, latent: Tensor) -> Tensor:
        """
        Calculate log probability of data point belonging to Gaussian
        distribution (i.e. p(c)). Formula follows the log of the
        gaussian PDF, simplified.

        Parameters
        ----------
        latent : Tensor
            Sampled latent vector.
        ----------

        Returns
        ----------
        Tensor
            Stacked tensors containing probabilities for each data point
            belonging in a cluster, for each cluster.
        ----------

        """
        # cluster means and variances
        mu_priors = self.mu_prior
        log_var_priors = self.log_var_prior

        log_probs = []

        # iterate over each cluster
        for (mean, log_var) in zip(mu_priors, log_var_priors):
            # print(f"latent shape: {latent.shape}")  # [128, 10]
            # print(f"mean shape: {mean.shape}")  # [10]
            # print(f"log var shape: {log_var.shape}")  # [10]
            log_prob = torch.pow(latent - mean, 2)
            # print(log_prob.shape)
            # [128, 10]
            log_prob += log_var
            log_prob += np.log(2 * np.pi)
            log_prob /= torch.exp(log_var)
            log_prob = -0.5 * torch.sum(log_prob, dim=1)
            # print(log_prob.shape)
            # [128]

            log_probs.append(log_prob.view(-1, 1))

        cat_shape = torch.cat(log_probs, 1)
        # print(cat_shape.shape)  # [128, 10]
        return cat_shape

    def elbo_loss(
        self,
        data: Tensor,
        reconstructed: Tensor,
        means: Tensor,
        log_vars: Tensor,
        latent: Tensor,
    ) -> Tensor:
        """ELBO loss function for variational deep embedded clustering (VaDE).

        Estimated lower bound (ELBO) loss function. Derivation can be found in
        the original Variational Deep Embedding (VaDE) Paper [Jiang et. al
        2016], with specific derivations found in the appendix.

        Parameters
        ----------
        data : Tensor
            Batch of input data.
        reconstructed : Tensor
            Batch of reconstructed input data, output by VAE.
        means : Tensor
            Vector containing mean of each latent variable.
        log_vars : Tensor
            Vector containing log variances of each latent variable.
        latent : Tensor
            Batch of sampled latent variable representations of input.
        ----------

        Returns
        ----------
        Tensor
            Loss value for given batch.
        ----------

        """

        # GMM variables
        pi_prior = self.pi_prior
        log_var_prior = self.log_var_prior
        mu_prior = self.mu_prior

        recon_loss = torch.nn.BCELoss()

        elbo = recon_loss(reconstructed, data) * data.size(1)

        # prob of z given c (p(z|c))
        log_probs = self.cluster_probabilities(latent)  # [128, 10]

        # print(pi_prior.unsqueeze(0).shape) [1, 10]

        # gamma numerator
        gamma = torch.exp(torch.log(pi_prior.unsqueeze(0)) + log_probs) + 1e-11

        # probability of c given x (q(c|x))
        # val = gamma / gamma.sum(dim=1).unsqueeze(1)
        gamma = gamma / gamma.sum(dim=1).unsqueeze(1)  # [128, 10]

        # print((val.sum(dim=1).shape))

        # first component of KL-divergence
        elbo += 0.5 * torch.mean(
            torch.sum(
                gamma
                * torch.sum(
                    log_var_prior.unsqueeze(0)
                    + torch.exp(log_vars.unsqueeze(1) - log_var_prior.unsqueeze(0))
                    + (
                        (means.unsqueeze(1) - mu_prior.unsqueeze(0)).pow(2)
                        / torch.exp(log_var_prior.unsqueeze(0))
                    ),
                    dim=2,
                ),
                dim=1,
            ),
        )

        # print(f"kld: {kld}")

        # second and third components of KL-divergence
        elbo -= torch.mean(torch.sum(gamma * torch.log(pi_prior.unsqueeze(0) / gamma), dim=1))
        elbo -= 0.5 * torch.mean(torch.sum(1 + log_vars, dim=1))

        return elbo

    def predict(self, latent: Tensor) -> np.ndarray:
        """
        Run VaDE model in inference mode on the latent projection of the input
        data. Returns the selected cluster for each point.

        Parameters
        ----------
        latent: torch.Tensor
            Sampled latent vectors for a given batch.
        ----------

        Returns
        ----------
        numpy.ndarray
            Cluster assignment for each data point>
        ----------

        """

        # GMM cluster definitions
        pi_prior = self.pi_prior

        # calculate individual probabilities
        cluster_probs = torch.exp(
            torch.log(pi_prior.unsqueeze(0)) + self.cluster_probabilities(latent)
        )

        # take largest probability for each point
        cluster_probs = cluster_probs.detach().cpu().numpy()

        max_prob = np.argmax(cluster_probs, axis=1)

        probs = cluster_probs / cluster_probs.sum(axis=1, keepdims=True)

        weights = probs.max(axis=1)

        # return largest probability cluster
        return max_prob, weights
