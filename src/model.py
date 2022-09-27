import os
from unittest import result
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # TODO 1: Define the encoder and decoder networks

        # TODO : Remove hard-coded values:hidden_dim, latent_dim, activation
        activation = nn.ReLU()
        hidden_channel_dim = 64
        latent_dim = 64*64*64

        # NOTE: that the input image is 256x256
        # TODO : Make the network work for any image size

        model = [
            ConvBlock(3,hidden_channel_dim,7,1,'same', activation),
            ConvBlock(hidden_channel_dim,hidden_channel_dim,4,2,1, activation),
            ConvBlock(hidden_channel_dim,hidden_channel_dim,4,2,1, activation),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock()
        ]
        self.encoder = nn.Sequential(*model)

        # After the encoder the size should be 64x64x64

        self.fc_mu = nn.Linear(hidden_channel_dim*64*64, latent_dim)
        self.fc_var = nn.Linear(hidden_channel_dim*64*64, latent_dim)

        model = [
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ConvBlock(64,64,4,2,1, activation),
            ConvBlock(64,64,4,2,1, activation),
            ConvBlock(64,3,7,1,'same', activation)
            ]
        self.decoder = nn.Sequential(*model)

        self.fc_decoder_input = nn.Linear(latent_dim, hidden_channel_dim*64*64)



    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        result = self.fc_decoder_input(result)
        result = result.view(result.size(0), 64, 64, 64)
        return self.decoder(result)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.ReLU()):
        model = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            activation]
        super(ConvBlock, self).__init__(*model)

class DeconvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.ReLU()):
        model = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            activation]
        super(DeconvBlock, self).__init__(*model)

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        # TODO : Define the ResBlock network structure
