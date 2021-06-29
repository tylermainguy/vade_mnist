"""Training and evaluation class for VAE clustering model."""
import os

import numpy as np
import seaborn as sns
import torch
from dotenv import load_dotenv
from sklearn.mixture import GaussianMixture
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.vade import VaDE
from utils import AverageMeter, cluster_acc, init_weights, remove_dropout

load_dotenv()


class Trainer:
    """
    Class for training and running inference on the variational deep embedding
    (VaDE) model for DESI-IMS data.
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

        # create model, init weights
        self.model = VaDE(args)
        self.model.apply(init_weights)
        self.model.to(self.device)

        self.dataloader = dataloader

        # for viewing model architecture
        print("-" * 60)
        print("Model Architecture")
        print("-" * 30)
        print("\n{}\n\n".format(self.model))
        print("-" * 60)

        # adam optimizer
        self.optimizer = Adam(self.model.parameters(), lr=args.lr)

        # gaussian mixture model for pretraining
        self.gmm = GaussianMixture(
            n_components=args.n_clusters, covariance_type="diag", random_state=0
        )

        # tensorboard logging
        self.writer = SummaryWriter()
        self.meter = AverageMeter()

        # model saving
        self.save_path = "ckpt/"

        # load existing model
        if self.args.load_model or self.args.eval:
            self.load_model()

        else:
            self.load_pretrained()

        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.95)

        # easier bug tracking
        if self.args.debug:
            torch.autograd.set_detect_anomaly(True)

    def load_pretrained(self):
        """Load pretrained VAE."""

        self.loss_idx = []
        self.loss_averages = []
        self.model.load_state_dict(
            torch.load(self.save_path + "pretrain.pk", map_location=self.device)
        )
        self.epoch = 0

    def load_model(self):
        """
        Load saved model data.
        """

        checkpoint = torch.load(self.save_path + "model.pk", map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.epoch = checkpoint["epoch"] + 1
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_averages = checkpoint["loss_averages"]
        self.loss_idx = checkpoint["loss_idx"]
        self.meter = checkpoint["average_meter"]

        del checkpoint

    def train(self):
        """
        Train the model for an epoch. Loss function is computed using the
        evidence lower bound (ELBO) approximation.

        Parameters
        ----------
        dataloader : torch.utils.data.dataloader.DataLoader
            DataLoader object containing training samples.
        epoch : int
            Current epoch number.
        ----------

        """

        self.model.train()

        # ignore dropout during clustering
        self.model.apply(remove_dropout)

        # batch iteration
        with tqdm(self.dataloader, unit="batches") as tepoch:
            tepoch.set_description("Epoch {}".format(self.epoch + 1))
            for batch_num, data in enumerate(tepoch):

                data, labels = data

                # mnist do be images though
                data = torch.flatten(data, start_dim=1)

                batch_size, _ = data.shape

                self.optimizer.zero_grad()

                # forward pass
                data = data.to(self.device)
                x_hat, latent, means, stds = self.model(data)

                # compute ELBO loss
                loss = self.model.elbo_loss(data, x_hat, means, stds, latent)

                self.meter.update(loss.item())

                # backprop
                loss.backward()
                self.optimizer.step()

                self.writer.add_scalar(
                    "Loss/train", self.meter.avg, self.epoch * len(self.dataloader) + batch_num
                )

                self.loss_idx.append(self.epoch * len(self.dataloader) + batch_num)
                self.loss_averages.append(self.meter.avg)

        # logging
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "epoch": self.epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_idx": self.loss_idx,
                "loss_averages": self.loss_averages,
                "average_meter": self.meter,
            },
            self.save_path + "model.pk",
        )
        print("\tepoch {} average loss: {}\n".format(self.epoch + 1, self.meter.avg))

    def evaluate(self):
        """Evaluate model accuracy"""

        self.model.eval()

        all_preds = []
        all_labels = []
        # batch iteration
        with tqdm(self.dataloader, unit="batches") as tepoch:
            tepoch.set_description("Epoch {}".format(self.epoch + 1))
            for batch_num, data in enumerate(tepoch):

                data, labels = data

                data = torch.flatten(data, start_dim=1)
                # forward pass
                data = data.to(self.device)
                _, latent, _, _ = self.model(data)

                preds, _ = self.model.predict(latent)

                preds = torch.tensor(preds)

                all_preds.append(preds)
                all_labels.append(labels)

        all_labels = torch.cat(all_labels).cpu().detach().numpy()
        all_preds = torch.cat(all_preds).cpu().detach().numpy()

        acc = cluster_acc(all_preds, all_labels) * 100

        print(f"Accuracy: {acc}")

    def training_loop(self):
        """
        Training loop for VAE clustering. Consists of model training on a
        shuffled dataset followed by visualization on an unshuffled dataset.
        """

        while self.epoch < self.args.epochs:
            if not self.args.eval:
                print("training vae...")
                self.train()

            self.evaluate()
            # self.visualize_tissue()

            if self.args.eval:
                return

            self.scheduler.step()

            self.epoch += 1
