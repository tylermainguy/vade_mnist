"""Utility functions for mass spectrometry preprocessing, and model."""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from matplotlib.patches import Ellipse
from scipy.io import loadmat
from sklearn.utils.linear_assignment_ import linear_assignment
from torch import Tensor, nn

load_dotenv()


class DataParallelWrapper(nn.DataParallel):
    """Wrapper for DataParallel that allows for model indexing."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def remove_dropout(layer):
    """Remove dropout layer from NN."""
    if isinstance(layer, nn.Dropout):
        layer.eval()


def init_weights(layer):
    """Initialize layer weights in a neural network."""

    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform(layer.weight)
        layer.bias.data.fill_(0.01)


class AverageMeter:
    """
    Tracking of average values used for loss and accuracy monitoring.
    Code taken from:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def model_params() -> dict:
    """Configure network and training parameters.

    Returns
    ----------
    dict
        Parameters for training and model construction.
    ----------

    """

    # init argument parser
    parser = argparse.ArgumentParser()

    # model and training options
    parser.add_argument("--length", type=int, default=None)
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--h1_size", type=int, default=500)
    parser.add_argument("--h2_size", type=int, default=2000)

    parser.add_argument(
        "--latent", type=int, default=10, help="number of latent distributions (default: 3)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="batch size used in training (default: 256)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs for model training (default: 5)"
    )
    parser.add_argument(
        "--pre_epochs", type=int, default=10, help="number of epochs to pretrain VAE (default: 2)"
    )
    parser.add_argument(
        "--n_clusters", type=int, default=10, help="number of clusters in GMM (default: 3)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-3, help="number of clusters in GMM (default: 3)"
    )
    parser.add_argument(
        "--pre_lr", type=float, default=0.001, help="number of clusters in GMM (default: 3)"
    )
    parser.add_argument("--debug", type=bool, default=False, help="set debug mode in torch")
    parser.add_argument(
        "--load_model",
        type=bool,
        default=False,
        const=True,
        nargs="?",
        help="load most recent model",
    )
    parser.add_argument(
        "--load_pretrain",
        type=bool,
        const=True,
        default=False,
        nargs="?",
        help="load most recent model pretraining",
    )
    parser.add_argument(
        "--eval",
        type=bool,
        const=True,
        nargs="?",
        default=False,
        help="evaluate model clustering on data",
    )
    parser.add_argument(
        "--device", default="cpu", help="select device to run model on (default: 'cpu')"
    )

    return parser.parse_args()


def cluster_acc(Y_pred, Y):
    assert Y_pred.size == Y.size

    D = max(Y_pred.max(), Y.max()) + 1

    w = np.zeros((D, D), dtype=np.int64)

    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1

    ind = linear_assignment(w.max() - w)

    return sum([w[i, j] for i, j in ind]) * 1.0 / Y_pred.size
