# Network modules used to build the models

from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize


class VAENetwork(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        latent_dim=64,
        activation=nn.ReLU,
        use_transpose_conv=True,
        interp_mode="nearest",
        upsampling_kernel_size=5,
    ):
        """
        Parameters
        ----------
        hidden_dim : int, optional
            Number of channels in the hidden layers, by default 64
        latent_dim : int, optional
            Number of channels in the latent layer, by default 64
        activation : nn.Module, optional
            Activation function, by default nn.ReLU
        use_transpose_conv : bool, optional
            Whether to use transpose convolution or resize convolution, by default True
        interp_mode : str, optional
            Interpolation mode for resize convolution, either "nearest" or "bilinear", by default "nearest"
        upsampling_kernel_size : int, optional
            Kernel size for resize convolution, by default 5
        """
        super(VAENetwork, self).__init__()

        activation = nn.ReLU(inplace=True)
        hidden_channel_dim = 64

        encoder = [
            ConvBlock(3, hidden_channel_dim, 7, 1, "same", activation),
            ConvBlock(hidden_channel_dim, hidden_channel_dim, 4, 2, 1, activation),
            ConvBlock(hidden_channel_dim, hidden_channel_dim, 4, 2, 1, activation),
            ResBlock(hidden_channel_dim),
            ResBlock(hidden_channel_dim),
            ResBlock(hidden_channel_dim),
            ResBlock(hidden_channel_dim),
        ]
        self.encoder = nn.Sequential(*encoder)

        decoder = [
            ResBlock(hidden_channel_dim),
            ResBlock(hidden_channel_dim),
            ResBlock(hidden_channel_dim),
            ResBlock(hidden_channel_dim),
        ]
        if use_transpose_conv:
            decoder += [
                DeconvBlock(64, 64, 4, 2, 1, activation),
                DeconvBlock(64, 64, 4, 2, 1, activation),
            ]
        else:
            decoder += [
                ResizeConvBlock(
                    64,
                    64,
                    upsampling_kernel_size,
                    1,
                    "same",
                    activation,
                    interp_mode=interp_mode,
                ),
                ResizeConvBlock(
                    64,
                    64,
                    upsampling_kernel_size,
                    1,
                    "same",
                    activation,
                    interp_mode=interp_mode,
                ),
            ]

        decoder += [
            nn.Conv2d(64, 3, 7, 1, "same"),
            nn.Tanh(),
        ]
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def encode_decode(self, x):
        return self.decode(self.encode(x))

    def forward(self, x):
        latent = self.encode(x)
        # Assume that the latent distributions are Gaussian with means = latent and variances = 1
        eps = torch.randn_like(latent)
        x_reconst = self.decode(latent + eps)  # Reparameterization trick
        return x_reconst, latent


# ConvBlock is a convolutional layer followed by a normalization layer(optional) and an activation function
class ConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        activation=nn.ReLU(inplace=True),
        use_norm=True,
    ):
        model = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
            )
        ]
        if use_norm:
            model += [nn.InstanceNorm2d(out_channels)]
        model += [activation]
        super(ConvBlock, self).__init__(*model)


# DeconvBlock is a transposed convolutional layer followed by a normalization layer(optional) and an activation function
class DeconvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        activation=nn.ReLU(inplace=True),
        use_norm=True,
    ):
        model = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
            )
        ]
        if use_norm:
            model += [nn.InstanceNorm2d(out_channels)]
        model += [activation]
        super(DeconvBlock, self).__init__(*model)


# Utility layer that performs the interpolation for a given scale factor
class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode="bilinear"):
        """
        Parameters
        ----------
        scale_factor : int
            Scale factor for interpolation, the output size is calculated as input_size * scale_factor
        mode : str, optional
            Interpolation mode, either "nearest" or "bilinear", by default "bilinear"
        """
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


# Upscaling layer that performs an upsampling followed by a convolutional layer
class ResizeConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        activation=nn.ReLU(inplace=True),
        use_norm=True,
        interp_mode="bilinear",
    ):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Kernel size for final convolution
        stride : int
            Stride for final convolution
        padding : int
            Padding for final convolution
        activation : nn.Module, optional
            Activation function, by default nn.ReLU(inplace=True)
        use_norm : bool, optional
            Whether to use a normalization layer, by default True
        interp_mode : str, optional
            Interpolation mode for upsampling, either "nearest" or "bilinear", by default "bilinear"
        """
        model = [
            Interpolate(2, interp_mode),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
            ),
        ]
        if use_norm:
            model += [nn.InstanceNorm2d(out_channels)]
        model += [activation]
        super(ResizeConvBlock, self).__init__(*model)


# ResBlock is a residual block with two convolutional layers
class ResBlock(nn.Module):
    def __init__(self, dim, activation=nn.ReLU(inplace=True)):
        super(ResBlock, self).__init__()
        model = [
            ConvBlock(dim, dim, 3, 1, "same", activation),
            ConvBlock(dim, dim, 3, 1, "same", activation),
        ]
        self.conv_block = nn.Sequential(*model)

    def forward(self, x):
        return x + self.conv_block(x)


# MappingNetwork links the latent spaces of the two vae models)
class MappingNetwork(nn.Module):
    def __init__(self):
        super(MappingNetwork, self).__init__()

        model_from_latent = [
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 512, 3, 1, 1),
        ]
        global_branch = [NonLocalBlock2d(512), ResBlock(512), ResBlock(512)]
        model_to_latent = [
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ConvBlock(512, 256, 3, 1, 1),
            ConvBlock(256, 128, 3, 1, 1),
            ConvBlock(128, 64, 3, 1, 1),
        ]
        self.model = nn.Sequential(*model_from_latent, *global_branch, *model_to_latent)

    def forward(self, x):
        return self.model(x)


# Implementation of Non-local block described in https://arxiv.org/abs/1711.07971
# Embedded gaussian version + ResBlocks after W
class NonLocalBlock2d(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock2d, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.phi = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.theta = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.w = nn.Conv2d(
            in_channels=self.inter_channels,
            out_channels=self.in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.resblock = nn.Sequential(
            ResBlock(self.in_channels),
            ResBlock(self.in_channels),
            ResBlock(self.in_channels),
        )

    def forward(self, x):
        g_x = self.g(x).view(x.size(0), self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)
        theta_x = self.theta(x).view(x.size(0), self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(x.size(0), self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)
        y = torch.matmul(f_div_C, g_x)

        w_y = self.w(
            y.permute(0, 2, 1)
            .contiguous()
            .view(x.size(0), self.inter_channels, *x.size()[2:])
        )
        z = self.resblock(w_y + x)
        return z


# MultiScaleDiscriminator downsamples the input image by a factor of 2 multiple times and apply a discriminator at each scale
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, n_scales=2, n_layers=4, in_channels=3):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [Discriminator(n_layers, in_channels) for _ in range(n_scales)]
        )

    def forward(self, x):
        res = []
        for net in self.discriminators:
            res.append(net(x))
            x = F.avg_pool2d(x, 2)
        return res


# Discriminator based on LSGAN discriminator
class Discriminator(nn.Module):
    def __init__(self, n_layers=4, in_channels=3):
        super(Discriminator, self).__init__()
        nf = 64
        model = [
            ConvBlock(
                in_channels,
                nf,
                4,
                2,
                1,
                activation=nn.LeakyReLU(0.2, inplace=True),
                use_norm=False,
            )
        ]
        for _ in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            model += [
                ConvBlock(
                    nf_prev, nf, 4, 2, 1, activation=nn.LeakyReLU(0.2, inplace=True)
                )
            ]

        model += [
            ConvBlock(nf, nf, 4, 1, 1, activation=nn.LeakyReLU(0.2, inplace=True)),
            nn.Conv2d(nf, 1, 4, 1, 1),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        res = [x]
        for layer in self.model:
            res.append(layer(res[-1]))
        return res


# VGG19 network for perceptual loss
class VGG19_torch(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19_torch, self).__init__()
        vgg_pretrained_features = models.vgg19(
            weights=models.VGG19_Weights.DEFAULT
        ).features
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


# Compute perceptual loss as weighted sum of L1 loss between VGG19 features at specified layers
class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19_torch()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# Compute GANLoss (MSE between the result of the discriminator and the target)
class GANLoss(nn.Module):
    def __call__(self, x, target_is_real):
        if isinstance(x[0], list):
            return self.multiscale_forward(x, target_is_real)
        else:
            return self.single_scale_forward(x, target_is_real)

    def multiscale_forward(self, x, target_is_real):
        # average loss over scales (loss are just summed in the original implementation)
        loss = 0
        n_scales = len(x)
        for i in range(n_scales):
            loss += self.single_scale_forward(x[i], target_is_real)
        return loss / n_scales

    def single_scale_forward(self, x, target_is_real):
        if isinstance(target_is_real, torch.Tensor):
            target_tensor = torch.ones_like(x[-1])
            target_tensor = torch.mul(
                target_tensor, target_is_real.unsqueeze(-1).unsqueeze(-1)
            )
        elif isinstance(target_is_real, bool):
            target_tensor = (
                torch.ones_like(x[-1]) if target_is_real else torch.zeros_like(x[-1])
            )
        else:
            raise ValueError(
                "target_is_real should be either Tensor or bool, but got {}".format(
                    type(target_is_real)
                )
            )
        return F.mse_loss(x[-1], target_tensor)


# Compute the feature loss between the discriminator features of the real and fake images
class DiscriminatorFeatureLoss:
    def __call__(self, x, y):
        if isinstance(x[0], list):
            return self.multiscale_forward(x, y)
        else:
            return self.single_scale_forward(x, y)

    def multiscale_forward(self, x, y):
        loss = 0
        for i in range(len(x)):
            loss += self.single_scale_forward(x[i], y[i])
        return loss / len(x)

    def single_scale_forward(self, x, y):
        loss = 0
        n_layers = len(x)
        for i in range(n_layers):
            # times 4 in the original repo
            loss += F.l1_loss(x[i], y[i])
        return loss / n_layers
