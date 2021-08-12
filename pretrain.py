"""Pretraing VAE using reconstruction loss."""

import gc
import itertools
import os

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from sklearn.mixture import GaussianMixture
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.decoder import Decoder
from models.encoder import Encoder
from models.vade import VaDE
from utils import AverageMeter, DataParallelWrapper, cluster_acc, init_weights

load_dotenv()


class Pretrain:
    """
    Class for pretraining a VaDE model for DESI-IMS data.
    """

    def __init__(self, args: dict, dataloader: DataLoader):
        """
        Parameters
        ----------
        args : dict
            Mapping between parameters and values.
        ----------
        """

        self.args = args
        self.device = args.device
        self.dataloader = dataloader

        # self.model = VaDE(args)
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)
        # self.encoder.apply(init_weights)
        # self.decoder.apply(init_weights)
        self.decoder = self.decoder.to(self.device)
        self.encoder = self.encoder.to(self.device)

        # adam optimizer
        self.optimizer = Adam(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()), lr=args.pre_lr
        )

        # gaussian mixture model for pretraining
        self.gmm = GaussianMixture(
            n_components=args.n_clusters, covariance_type="diag", random_state=0
        )

        # model saving
        self.save_path = "ckpt/"

        # tensorboard logging
        self.writer = SummaryWriter()

        self.meter = AverageMeter()

        # tracking bugs in backprop
        if self.args.debug:
            torch.autograd.set_detect_anomaly(True)

        self.loss_idx = []
        self.loss_averages = []

    def train_recon(self):
        """
        Train the reconstruction loss of the VAE model as a pretraining step
        before clustering. For details regarding the usefulness of this for
        AE and VAE-based clusterings, refer to the paper Deep Embedded
        Clustering (Xie et al. 2016) here: https://arxiv.org/abs/1511.06335.
        """

        self.encoder.train()
        self.decoder.train()

        print("-" * 60)
        print("pretraining vae...")
        print("-" * 30)

        recon_loss = torch.nn.MSELoss()

        # epoch iteration
        for epoch in range(self.args.pre_epochs):
            # batch iteration

            self.meter.reset()
            with tqdm(self.dataloader, desc="pretraining", unit="batches") as tepoch:
                tepoch.set_description("Epoch {}".format(epoch + 1))
                for batch_num, data in enumerate(tepoch):

                    # extract
                    data, labels = data
                    data = torch.flatten(data, start_dim=1)
                    batch_size, _ = data.shape

                    data = data.to(self.device)

                    self.optimizer.zero_grad()

                    # get reconstruction
                    [means, _] = self.encoder(data)
                    x_hat = self.decoder(means)

                    # compute BCE loss
                    loss = recon_loss(data, x_hat)

                    # backprop
                    loss.backward()
                    self.optimizer.step()

                    self.meter.update(loss.detach().cpu().numpy())

                print("\tepoch {} average loss: {}\n".format(epoch, self.meter.avg))

        # # collect garbage, empty cache (cuda memory issues)
        # gc.collect()
        # torch.cuda.empty_cache()

    def fit_gmm(self):
        """
        Fit a gaussian mixture model (GMM) to IMS data embedded in the latent
        space of the VAE. The resulting cluster distributions will be used
        as initial values for the GMM used by the VAE model.
        """

        all_data = []
        all_labels = []

        # eval mode for latent variable generation
        self.encoder.eval()

        # shouldn't update network for inference
        with tqdm(self.dataloader, desc="GMM fitting", unit="batches") as tepoch:
            for data, labels in tepoch:

                data = torch.flatten(data, start_dim=1)
                data = data.to(self.device)

                # compress data into latent
                with torch.no_grad():
                    means, _ = self.encoder(data)

                all_data.append(means)
                all_labels.append(labels)

        all_labels = torch.cat(all_labels).cpu().detach().numpy()
        all_latents = torch.cat(all_data)
        all_data = all_latents.cpu().detach().numpy()

        # fit GMM on batch
        predicted = self.gmm.fit_predict(all_data)

        acc = cluster_acc(predicted, all_labels)

        print(f"Accuracy: {acc*100}%")

        return (all_latents, all_labels)

    def pretrain(self):
        """
        Pretrain autoencoders before clustering. Saves pretrained model for
        use in future epochs.

        Parameters
        ----------
        dataloader : DataLoader
            Dataloader object for all IMS samples.
        ----------

        """

        # train reconstruction of VAE
        self.train_recon()

        # fit GMM on encoded data
        all_data, all_labels = self.fit_gmm()

        # output the original GMM cluster weights
        print("weights: {}".format(self.gmm.weights_))

        self.model = VaDE(self.args, self.encoder, self.decoder)
        self.model.to(self.device)

        # update GMM init with GMM ran on pretrained autoncoder output
        self.model.pi_prior.data = torch.from_numpy(self.gmm.weights_).float().to(self.device)
        self.model.mu_prior.data = torch.from_numpy(self.gmm.means_).float().to(self.device)

        self.model.log_var_prior.data = (
            torch.log(torch.from_numpy(self.gmm.covariances_)).float().to(self.device)
        )

        predicted, _ = self.model.predict(all_data)

        acc = cluster_acc(predicted, all_labels)

        print(f"VaDE acc: {acc}")

        # save pretrained model
        torch.save(self.model.state_dict(), self.save_path + "pretrain.pk")

        print("-" * 30)
        print("pretraining complete...")
        print("-" * 60)
