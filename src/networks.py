# Network modules used to build the models

from torchvision import models
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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation=nn.ReLU(), use_norm=True):
        model = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
            )]
        if use_norm:
            model += [nn.InstanceNorm2d(out_channels)]
        model += [activation]
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
            nn.InstanceNorm2d(out_channels),
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


class MappingNetwork(nn.Module):
    def __init__(self):
        super(MappingNetwork, self).__init__()

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
        self.resblock = nn.Sequential(ResBlock(self.in_channels),
                                      ResBlock(self.in_channels),
                                      ResBlock(self.in_channels))

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


class Discriminator(nn.Module):
    def __init__(self, n_layer=4, in_channels=3):
        super(Discriminator, self).__init__()
        # TODO : Verify the network structure in the paper
        nf = 64
        model = [
            ConvBlock(in_channels, nf, 4, 2, 1, activation=nn.LeakyReLU(0.2), use_norm=False)]
        for _ in range(1, n_layer):
            nf_prev = nf
            nf = min(nf * 2, 512)
            model += [ConvBlock(nf_prev, nf, 4, 2, 1,
                                activation=nn.LeakyReLU(0.2))]

        model += [ConvBlock(nf, nf, 4, 1, 1, activation=nn.LeakyReLU(0.2)),
                  nn.Conv2d(nf, 1, 4, 1, 1)
                  ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        res = [x]
        for layer in self.model:
            res.append(layer(res[-1]))
        return res


class VGG19_torch(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19_torch, self).__init__()
        vgg_pretrained_features = models.vgg19(
            weights=models.VGG19_Weights.DEFAULT).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss_torch(nn.Module):
    def __init__(self):
        super(VGGLoss_torch, self).__init__()
        self.vgg = VGG19_torch()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class GANLoss(nn.Module):
    def forward(self, x, target_is_real):
        if isinstance(target_is_real, torch.Tensor):
            target_tensor = torch.ones_like(x)
            target_tensor = torch.mul(
                target_tensor, target_is_real.unsqueeze(-1).unsqueeze(-1))
        elif isinstance(target_is_real, bool):
            target_tensor = torch.ones_like(
                x) if target_is_real else torch.zeros_like(x)
        else:
            raise ValueError(
                "target_is_real should be either Tensor or bool, but got {}".format(type(target_is_real)))
        return F.mse_loss(x, target_tensor)
