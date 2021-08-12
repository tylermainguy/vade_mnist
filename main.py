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
from utils import model_params

load_dotenv()


def main():
    """
    Main function for training and evaluating the model.

    Handles the pretraining, training, and evaluating of the VAE model used to
    cluster IMS data.

    """

    # deterministic outputs
    # torch.manual_seed(0)
    # np.random.seed(0)

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

    # print(mnist_train[0])
    full_data = torch.utils.data.ConcatDataset([mnist_train, mnist_test])

    train_loader = DataLoader(full_data, batch_size=800, shuffle=True, num_workers=8)

    # if no user specified input size
    if not args.input_size:
        args.input_size = 784

    if not args.length:
        args.length = len(full_data)
        print(args.length)

    # pretrain reconstruction
    if not args.load_pretrain and not args.eval and not args.load_model:
        Pretrain(args, train_loader).pretrain()

    # # initialize training module with network parameters
    trainer = Trainer(args, train_loader)
    trainer.training_loop()


if __name__ == "__main__":
    main()
