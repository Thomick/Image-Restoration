# Network modules used to build the models

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # TODO : Remove hard-coded values:hidden_dim, activation
        activation = nn.ReLU()
        hidden_channel_dim = 64

        model = [
            ConvBlock(3, hidden_channel_dim, 7, 1, 'same', activation),
            ConvBlock(hidden_channel_dim, hidden_channel_dim,
                      4, 2, 1, activation),
            ConvBlock(hidden_channel_dim, hidden_channel_dim,
                      4, 2, 1, activation),
            ResBlock(hidden_channel_dim),
            ResBlock(hidden_channel_dim),
            ResBlock(hidden_channel_dim),
            ResBlock(hidden_channel_dim)
        ]
        self.encoder = nn.Sequential(*model)

        model = [
            ResBlock(hidden_channel_dim),
            ResBlock(hidden_channel_dim),
            ResBlock(hidden_channel_dim),
            ResBlock(hidden_channel_dim),
            DeconvBlock(64, 64, 4, 2, 1, activation),
            DeconvBlock(64, 64, 4, 2, 1, activation),
            nn.Conv2d(64, 3, 7, 1, 'same')
        ]
        self.decoder = nn.Sequential(*model)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        latent = self.encode(x)
        # Assume that the latent distributions are Gaussian with means = latent and variances = 1
        eps = torch.randn_like(latent)
        x_reconst = self.decode(latent+eps)  # Reparameterization trick
        return x_reconst, latent


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
    def __init__(self, dim, activation=nn.ReLU()):
        super(ResBlock, self).__init__()
        # TODO : Verify the ResBlock network structure
        model = [
            ConvBlock(dim, dim, 3, 1, 'same', activation),
            ConvBlock(dim, dim, 3, 1, 'same', activation)
        ]
        self.conv_block = nn.Sequential(*model)

    def forward(self, x):
        return x + self.conv_block(x)


class Mapping(nn.Module):
    def __init__(self):
        super(Mapping, self).__init__()

        model_from_latent = [
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 512, 3, 1, 1),
        ]
        global_branch = [
            NonLocalBlock2d(512),
            ResBlock(512),
            ResBlock(512)]
        model_to_latent = [
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ConvBlock(512, 256, 3, 1, 1),
            ConvBlock(256, 128, 3, 1, 1),
            ConvBlock(128, 64, 3, 1, 1)]
        self.model = nn.Sequential(
            *model_from_latent, *global_branch, *model_to_latent)

    def forward(self, x):
        return self.model(x)


class NonLocalBlock2d(nn.Module):
    # Implementation of Non-local block from https://arxiv.org/abs/1711.07971
    # Embedded gaussian version + ResBlocks after W
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock2d, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(in_channels=self.in_channels,
                           out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels,
                             out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.w = nn.Conv2d(in_channels=self.inter_channels,
                           out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        self.resblock = nn.Sequential([ResBlock(self.in_channels),
                                      ResBlock(self.in_channels),
                                      ResBlock(self.in_channels)])

    def forward(self, x):
        g_x = self.g(x).view(x.size(0), self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(x.size(0), self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(x.size(0), self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)

        w_y = self.w(y.permute(0, 2, 1).contiguous().view(
            x.size(0), self.inter_channels, *x.size()[2:]))
        z = self.resblock(w_y + x)
        return z


def SN(module): return torch.nn.utils.spectral_norm(module)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # TODO : Verify the network structure in the paper
        model = [
            SN(nn.Conv2d(3, 64, 4, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False),
            SN(nn.Conv2d(64, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False),
            SN(nn.Conv2d(64, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False),
            SN(nn.Conv2d(64, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False),
            SN(nn.Conv2d(64, 1, 3, 2, 1)),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x).reshape(-1, 1)
