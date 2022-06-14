# Benchmark

Benchmarks were run on RTX 3090 GPU. 

## Tech stack
- Python 3.9.7
- CUDA 11.3

## Set the environment up
- Linux
```bash
python -m venv benchmark_env
source ./benchmark_env/bin/activate
```

- Windows
```bash
py -3.9 -m venv benchmark_env
.\benchmark_env\Scripts\activate.bat
```

## Install dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

## Traning

### Download data

- Datasets for CNN and ACGAN are downloaded automatically to `data` folder when training script is launched for the first time. 
- EffNet requires ImageNet dataset. You can download Tiny ImaegNet dataset on [Kaggle](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet). Place `train`, `val` and `test` folders to `data/tiny_imagenet` folder.

### Traning scripts
- `--epochs` and `--batch_size` arguments specify the number of epochs/batch size used for the traning.
- There are distinct scripts to train every model.
```bash
python -m scripts.cnn --epochs 2 --batch_size 128
# or
python -m scripts.acgan --epochs 2 --batch_size 128
# or
python -m scripts.effnet --epochs 2 --batch_size 128
```

- Logs are saved to `logs` folder. Metrics are saved to `logs/tensorboard`, profiling logs are saved to `logs/profiler`.

## GUI for logs
- Metrics: ```tensorboard --logdir logs/tensorboard```
- Profiling: ```tensorboard --logdir logs/profiler```

## Project structure
```bash
.
├── benchmark
│   ├── __init__.py
│   ├── datamodules.py  # PyTorch Lightning data modules
│   └── models  # module with NN implementations
├── data  # folder containing datasets
│   ├── cifar100
│   ├── mnist
│   └── tiny_imagenet
├── logs
│   ├── profiler  # profiler logs
│   └── tensorboard  # training logs
├── scripts  # scripts to run the respective models
│   ├── acgan.py
│   ├── cnn.py
│   └── effnet.py
├── README.md
├── requirements.txt
└── setup.py
```