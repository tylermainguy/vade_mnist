"""Main function for training mass spectrometry clustering using VAEs."""

import os
import warnings

import numpy as np
import torch
import torchvision.datasets as datasets
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision import transforms

from pretrain import Pretrain
from trainer import Trainer
from utils import get_image_dims, model_params

load_dotenv()


def main():
    """
    Main function for training and evaluating the model.

    Handles the pretraining, training, and evaluating of the VAE model used to
    cluster IMS data.

    """


    # deterministic outputs
    torch.manual_seed(0)
    np.random.seed(0)

    args = model_params()

    # use gpus when available
    if torch.cuda.is_available():
        args.device = "cuda"

    mnist_train = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    mnist_test = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)

    # if no user specified input size
    if not args.input_size:
        args.input_size = 784

    if not args.length:
        args.length = len(mnist_train)

    # pretrain reconstruction
    if not args.load_pretrain and not args.eval and not args.load_model:
        Pretrain(args, train_loader).pretrain()

    # # initialize training module with network parameters
    trainer = Trainer(args, train_loader)
    trainer.training_loop()


if __name__ == "__main__":
    main()
