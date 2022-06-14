from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import PyTorchProfiler

from benchmark.argparser import parse_args
from benchmark.datamodules import TinyImageNetDataModule
from benchmark.models.effnet import LitEffNet


def main():
    args = parse_args()

    mnist = TinyImageNetDataModule(batch_size=args.batch_size)
    model = LitEffNet()

    logger = TensorBoardLogger('logs/tensorboard', name='effnet_benchmark')

    profiler = PyTorchProfiler(
        dirpath=f'logs/profiler/effnet_e_{args.epochs}_b_{args.batch_size}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
        filename='effnet_benchmark'
    )
    trainer = pl.Trainer(logger=logger, max_epochs=args.epochs, gpus=1, profiler=profiler)

    trainer.fit(model, datamodule=mnist)
    trainer.test(model, datamodule=mnist)


if __name__ == '__main__':
    main()
