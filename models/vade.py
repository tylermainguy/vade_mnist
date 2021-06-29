"""Module for Vade."""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from models.decoder import Decoder
from models.encoder import Encoder


class VaDE(nn.Module):
    """VaDE clustering model."""

    def __init__(self, args: dict):
        """
        Parameters
        ----------
        args : dict
            Mapping containing parameters for network initialization.
        ----------

        """
        super().__init__()

        # initialiaze symmetric encoder-decoder
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

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
        latent, means, log_vars = self.encoder(data)

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
            log_prob = torch.pow(latent - mean, 2) / torch.exp(log_var)
            log_prob += log_var
            log_prob += np.log(2 * np.pi)
            log_prob = -0.5 * torch.sum(log_prob, dim=1)

            log_probs.append(log_prob.view(-1, 1))

        return torch.cat(log_probs, 1)

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

        # reconstruction loss
        elbo = recon_loss(reconstructed, data) * data.size(1)

        # prob of z given c (p(z|c))
        log_probs = self.cluster_probabilities(latent)

        # gamma numerator
        gamma = torch.exp(torch.log(pi_prior.unsqueeze(0)) + log_probs) + 1e-11

        # probability of c given x (q(c|x))
        gamma = gamma / gamma.sum(dim=1).unsqueeze(1)

        # first component of KL-divergence
        kld = 0.5 * torch.mean(
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

        # second and third components of KL-divergence
        kld -= torch.mean(torch.sum(gamma * torch.log(pi_prior.unsqueeze(0) / gamma), dim=1))
        kld -= 0.5 * torch.mean(torch.sum(1 + log_vars, dim=1))

        # add KL to ELBO loss
        elbo += kld

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
