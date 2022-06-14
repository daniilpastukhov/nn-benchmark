import torch
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl


class LitEffNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.efficientnet_b2()

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
        x = batch
        output = self(x)
        pred = output.argmax(dim=1)
        # accuracy = torchmetrics.functional.accuracy(pred, y)
        # self.log('test/accuracy', accuracy)
        # return accuracy

    def forward(self, x):
        return self.model(x)
