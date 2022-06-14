from typing import Tuple, Dict
import wandb

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchmetrics
from torchvision.utils import make_grid
from torchgan.models import ACGANGenerator, ACGANDiscriminator
import pytorch_lightning as pl


class LitACGAN(pl.LightningModule):
    def __init__(self,
                 num_classes: int,
                 img_size: int,
                 out_channels: int = 1,
                 step_channels: int = 32,
                 latent_dim: int = 100,
                 lr: float = 0.0002,
                 betas: Tuple[float, float] = (0.5, 0.999),
                 transforms: nn.Module = nn.Identity()) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False
        self._latent_dim = latent_dim
        self.img_size = img_size
        self.num_classes = num_classes
        self.threshold = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.transforms = transforms

        self._generator = ACGANGenerator(num_classes=num_classes,
                                         encoding_dims=latent_dim,
                                         out_size=img_size,
                                         out_channels=out_channels,
                                         step_channels=step_channels)
        self._discriminator = ACGANDiscriminator(num_classes=num_classes,
                                                 in_size=img_size,
                                                 in_channels=out_channels,
                                                 step_channels=step_channels,
                                                 last_nonlinearity=nn.Sigmoid())

        val_n = 3
        self._val_z = torch.randn(val_n * num_classes, self._latent_dim)
        self._val_labels = torch.arange(num_classes).unsqueeze(dim=1).expand(num_classes, val_n).flatten()

    def forward(self, x, y):
        return self._generator(x, y)

    def predict(self, x):
        x = self.transforms(x)
        return self._discriminator(x, mode=None)

    def predict_label(self, x):
        x = self.transforms(x)
        return self._discriminator(x, mode='classifier')

    def predict_outlier_score(self, x):
        x = self.transforms(x)
        return self._discriminator(x, mode='discriminator')

    def configure_optimizers(self):
        lr = self.hparams.lr
        betas = self.hparams.betas

        opt_g = optim.Adam(self._generator.parameters(), lr=lr, betas=betas)
        opt_d = optim.Adam(self._discriminator.parameters(), lr=lr, betas=betas)
        return opt_g, opt_d

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        inputs, labels = batch
        inputs = self.transforms(inputs)
        inputs = inputs + 0.1 * torch.randn_like(inputs)
        labels = labels.long()

        ones = torch.ones(len(inputs), device=self.device, dtype=torch.float)
        zeros = torch.zeros(len(inputs), device=self.device, dtype=torch.float)

        noise = torch.randn(len(inputs), self._latent_dim, device=self.device)
        fake_labels = torch.randint_like(labels, low=0, high=self.num_classes, device=self.device)
        fake_inputs = self._generator(noise, fake_labels)

        def train_generator():
            self._generator.zero_grad()
            dx, aux = self._discriminator(fake_inputs, mode=None)
            g_loss_dx = F.binary_cross_entropy(dx, ones)
            self.manual_backward(g_loss_dx, retain_graph=True)
            g_loss_aux = F.cross_entropy(aux, fake_labels)
            self.manual_backward(g_loss_aux)
            opt_g.step()
            self.log('train/g_loss_dx', g_loss_dx, on_epoch=True, on_step=False)
            self.log('train/g_loss_aux', g_loss_aux, on_epoch=True, on_step=False)

        def train_discriminator():
            self._discriminator.zero_grad()
            dx, aux = self._discriminator(inputs, mode=None)
            d_loss_dx = F.binary_cross_entropy(dx, ones if self.current_epoch % 5 else zeros)
            self.manual_backward(d_loss_dx, retain_graph=True)
            d_loss_aux = F.cross_entropy(aux, labels)
            self.manual_backward(d_loss_aux)
            d_real_loss_dx = d_loss_dx.item()
            d_real_loss_aux = d_loss_aux.item()

            dx = self._discriminator(fake_inputs.detach(), mode='discriminator')
            d_loss_dx = F.binary_cross_entropy(dx, zeros if self.current_epoch % 5 else ones)
            self.manual_backward(d_loss_dx)
            d_fake_loss_dx = d_loss_dx.item()
            opt_d.step()
            self.log('train/d_real_loss_dx', d_real_loss_dx, on_epoch=True, on_step=False)
            self.log('train/d_real_loss_aux', d_real_loss_aux, on_epoch=True, on_step=False)
            self.log('train/d_fake_loss_dx', d_fake_loss_dx, on_epoch=True, on_step=False)

        train_discriminator()
        train_generator()

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = self.transforms(inputs)
        scores, predictions = self._discriminator(inputs, mode=None)
        self.threshold = nn.Parameter(torch.quantile(scores, 0.05), requires_grad=False)
        od_preds = (scores > self.threshold).int()
        f1 = torchmetrics.functional.f1_score(od_preds, labels, average='macro', num_classes=self.num_classes)
        accuracy = torchmetrics.functional.accuracy(od_preds, torch.ones_like(od_preds))
        self.log('val/f1', f1, on_epoch=True, on_step=False)
        self.log('val/accuracy', accuracy, on_epoch=True, on_step=False)
        return f1, accuracy

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs = self.transforms(inputs)
        predictions = self._discriminator(inputs, mode='classifier')
        f1 = torchmetrics.functional.f1_score(predictions, labels, average='macro', num_classes=self.num_classes)
        self.log('test/f1', f1, on_epoch=True, on_step=False)
        return f1

    def on_fit_start(self):
        self._val_z = self._val_z.to(self.device)
        self._val_labels = self._val_labels.to(self.device)
        self.threshold = self.threshold.to(self.device)

    @torch.no_grad()
    def on_epoch_end(self):
        if self.current_epoch % 20 == 0:
            self._generator.eval()
            # log sampled images
            sample_imgs = self._generator(self._val_z, self._val_labels)
            grid = make_grid(sample_imgs, nrow=3, normalize=True)
            # self.logger.experiment.log(
            #     {'generated_images': wandb.Image(grid, caption='Generated images')},
            #     step=self.global_step
            # )
