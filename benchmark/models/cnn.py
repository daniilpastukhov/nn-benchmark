import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class LitCNN(pl.LightningModule):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.model = CNN(n_channels, n_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        return optimizer

    def common_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('train/loss', loss)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('val/loss', loss)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        pred = output.argmax(dim=1)
        accuracy = torchmetrics.functional.accuracy(pred, y)
        self.log('test/accuracy', accuracy)
        return accuracy

    def forward(self, x):
        return self.model(x)


class CNN(nn.Module):
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels

        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 8, kernel_size=(5, 5), stride=(1, 1)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=(5, 5), stride=(1, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.LeakyReLU(),
            nn.LazyLinear(n_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
