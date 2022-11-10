# Actual models assembled using the network modules defined in networks.py

import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from networks import (
    VAENetwork,
    MappingNetwork,
    Discriminator,
    VGGLoss,
    GANLoss,
    DiscriminatorFeatureLoss,
)
from utils import save_image


# TODO : Use Multi-Scale Discriminator and add specific parameters for it

# VAE with interwined latent space for real and synthetic images
# Input: Noisy images (real and synthetic)
# Output: Reconstructed noisy images
class VAE1(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.vae = VAENetwork()
        self.discriminator = Discriminator(4)
        self.discriminator_latent = Discriminator(3, in_channels=64)
        self.params = params
        self.curr_device = None
        self.loss_vgg = VGGLoss()
        self.loss_gan = GANLoss()
        self.loss_feat_gan = DiscriminatorFeatureLoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.curr_device = batch[0].device

        input_img, label = batch  # 1=real noise, 0=generated noise
        reconst_img, latent = self(input_img)
        label = label.type_as(input_img).reshape(-1, 1)

        # train VAE1
        if optimizer_idx == 0:

            pred_disc_real = self.discriminator(input_img)
            pred_disc_fake = self.discriminator(reconst_img)
            loss_g_gan = self.loss_gan(pred_disc_fake, target_is_real=True)
            loss_kl = torch.mean(torch.pow(latent, 2)) / 2
            loss_reconst = F.l1_loss(reconst_img, input_img)
            loss_latent_gan = self.loss_gan(
                self.discriminator_latent(latent), 1 - label
            )

            loss_feat_gan = self.loss_feat_gan(pred_disc_real, pred_disc_fake)

            loss_vgg = self.loss_vgg(reconst_img, input_img)
            vae_loss = (
                loss_kl
                + self.params["a_reconst"] * loss_reconst
                + loss_g_gan
                + loss_latent_gan
                + (loss_vgg + loss_feat_gan) * self.params["lambda2_feat"]
            )

            self.log("vae1_loss", vae_loss, prog_bar=True)
            self.log("loss_g_gan", loss_g_gan)
            self.log("loss_latent_gan", loss_latent_gan)
            self.log("loss_kl", loss_kl)
            self.log("loss_reconst", loss_reconst)
            self.log("loss_feat_gan", loss_feat_gan)
            self.log("loss_vgg", loss_vgg)
            return vae_loss

        # train discriminator
        if optimizer_idx == 1:

            real_loss = self.loss_gan(
                self.discriminator(input_img), target_is_real=True
            )

            fake_loss = self.loss_gan(
                self.discriminator(reconst_img), target_is_real=False
            )

            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

        # train discriminator for the latent space
        if optimizer_idx == 2:

            d_latent_loss = self.loss_gan(self.discriminator_latent(latent), label)

            self.log("d_latent_loss", d_latent_loss, prog_bar=True)
            return d_latent_loss

    def configure_optimizers(self):
        lr = self.params["lr"]
        b1 = self.params["b1"]
        b2 = self.params["b2"]

        opt_vae = torch.optim.Adam(self.vae.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        opt_d_latent = torch.optim.Adam(
            self.discriminator_latent.parameters(), lr=lr, betas=(b1, b2)
        )
        return [opt_vae, opt_d, opt_d_latent], []

    def validation_step(self, batch, batch_idx):
        self.curr_device = batch[0].device
        # TODO: Implement the validation step for VAE1
        pass

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, _ = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)

        # test_input, test_label = batch
        recons = self.vae.decode(self.vae.encode(test_input))
        save_image(
            recons.data,
            os.path.join(
                self.logger.log_dir,
                "Reconstructions",
                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png",
            ),
        )
        save_image(
            test_input.data,
            os.path.join(
                self.logger.log_dir,
                "Input",
                f"input_{self.logger.name}_Epoch_{self.current_epoch}.png",
            ),
        )


# Classic VAE
# Input: Clean image
# Output: Reconstructed clean image
class VAE2(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.vae = VAENetwork()
        self.discriminator = Discriminator()
        self.params = params
        self.curr_device = None
        self.loss_vgg = VGGLoss()
        self.loss_gan = GANLoss()
        self.loss_feat_gan = DiscriminatorFeatureLoss()
        self.save_hyperparameters()

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.curr_device = batch[0].device

        real_img, _ = batch
        reconst_img, latent = self(real_img)

        # train VAE2
        if optimizer_idx == 0:

            pred_disc_real = self.discriminator(real_img)
            pred_disc_fake = self.discriminator(reconst_img)
            loss_g_gan = self.loss_gan(pred_disc_fake, target_is_real=True)

            loss_kl = torch.mean(torch.pow(latent, 2)) / 2
            loss_reconst = F.l1_loss(reconst_img, real_img)
            loss_feat_gan = self.loss_feat_gan(pred_disc_real, pred_disc_fake)
            loss_vgg = self.loss_vgg(reconst_img, real_img)
            vae_loss = (
                loss_kl
                + self.params["a_reconst"] * loss_reconst
                + loss_g_gan
                + (loss_vgg + loss_feat_gan) * self.params["lambda2_feat"]
            )

            self.log("vae2_loss", vae_loss, prog_bar=True)
            self.log("loss_g_gan", loss_g_gan)
            self.log("loss_kl", loss_kl)
            self.log("loss_reconst", loss_reconst)
            self.log("loss_feat_gan", loss_feat_gan)
            self.log("loss_vgg", loss_vgg)
            return vae_loss

        # train discriminator to distinguish real and reconstructed images (1=real, 0=recoonstructed)
        if optimizer_idx == 1:
            real_loss = self.loss_gan(self.discriminator(real_img), target_is_real=True)

            fake_loss = self.loss_gan(
                self.discriminator(reconst_img), target_is_real=False
            )

            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        lr = self.params["lr"]
        b1 = self.params["b1"]
        b2 = self.params["b2"]

        opt_vae = torch.optim.Adam(self.vae.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_vae, opt_d], []

    def validation_step(self, batch, batch_idx):
        self.curr_device = batch[0].device
        # TODO: Implement the validation step for VAE2
        pass

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, _ = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)

        # test_input, test_label = batch
        recons = self.vae.decode(self.vae.encode(test_input))
        save_image(
            recons.data,
            os.path.join(
                self.logger.log_dir,
                "Reconstructions",
                f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png",
            ),
        )
        save_image(
            test_input.data,
            os.path.join(
                self.logger.log_dir,
                "Input",
                f"input_{self.logger.name}_Epoch_{self.current_epoch}.png",
            ),
        )


# Mapping from the latent space of VAE1 to the latent space of VAE2
class Mapping(pl.LightningModule):
    def __init__(self, params, vae1_encoder, vae2):
        super().__init__()
        self.vae1_encoder = vae1_encoder
        self.vae2 = vae2
        self.mapping = MappingNetwork()
        self.discriminator = Discriminator()
        self.params = params
        self.loss_vgg = VGGLoss()
        self.loss_gan = GANLoss()
        self.loss_feat_gan = DiscriminatorFeatureLoss()
        self.save_hyperparameters(ignore=["vae1_encoder", "vae2"])

    def forward(self, x):
        latent1 = self.vae1_encoder(x)
        latent2 = self.mapping(latent1)
        return self.vae2.decode(latent2), latent1, latent2

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.curr_device = batch[0].device

        imgs, _ = batch
        clean_img = imgs[:, 0, :, :, :]
        noisy_img = imgs[:, 1, :, :, :]
        if optimizer_idx == 0:
            denoised, _, latent_denoised = self(noisy_img)
            latent_clean = self.vae2.encode(clean_img)
            restored = self.vae2.decode(latent_clean)

            latent_loss = F.l1_loss(latent_denoised, latent_clean)

            pred_disc_real = self.discriminator(restored)
            pred_disc_fake = self.discriminator(denoised)
            loss_g_gan = self.loss_gan(pred_disc_fake, target_is_real=True)
            loss_feat_gan = self.loss_feat_gan(pred_disc_real, pred_disc_fake)

            loss_vgg = self.loss_vgg(denoised, restored)
            mapping_loss = (
                self.params["lambda1_recons"] * latent_loss
                + loss_g_gan
                + (loss_vgg + loss_feat_gan) * self.params["lambda2_feat"]
            )
            self.log("latent_loss", latent_loss)
            self.log("loss_g_gan", loss_g_gan)
            self.log("mapping_loss", mapping_loss, prog_bar=True)
            self.log("loss_feat_gan", loss_feat_gan)
            self.log("loss_vgg", loss_vgg)
            return mapping_loss

        # train discriminator to distinguish real and reconstructed images (1=real, 0=recoonstructed)
        if optimizer_idx == 1:

            denoised, _, _ = self(noisy_img)
            restored = self.vae2.decode(self.vae2.encode(clean_img))

            real_loss = self.loss_gan(self.discriminator(restored), True)
            fake_loss = self.loss_gan(self.discriminator(denoised), False)

            d_loss = (real_loss + fake_loss) / 2
            self.log("d_loss", d_loss, prog_bar=True)
            return d_loss

    def configure_optimizers(self):
        lr = self.params["lr"]
        b1 = self.params["b1"]
        b2 = self.params["b2"]

        opt_mapping = torch.optim.Adam(self.mapping.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_mapping, opt_d], []

    def validation_step(self, batch, batch_idx):
        self.curr_device = batch[0].device
        pass

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, _ = next(iter(self.trainer.datamodule.test_dataloader()))
        clean_input = test_input[:, 0, :, :, :].to(self.curr_device)
        noisy_input = test_input[:, 1, :, :, :].to(self.curr_device)

        # test_input, test_label = batch
        recons, _, _ = self(noisy_input)
        save_image(
            recons.data,
            os.path.join(
                self.logger.log_dir,
                "Results",
                f"result_{self.logger.name}_Epoch_{self.current_epoch}.png",
            ),
        )
        save_image(
            clean_input.data,
            os.path.join(
                self.logger.log_dir,
                "Clean_input",
                f"clean_{self.logger.name}_Epoch_{self.current_epoch}.png",
            ),
        )
        save_image(
            noisy_input.data,
            os.path.join(
                self.logger.log_dir,
                "Noisy_input",
                f"noisy_{self.logger.name}_Epoch_{self.current_epoch}.png",
            ),
        )
