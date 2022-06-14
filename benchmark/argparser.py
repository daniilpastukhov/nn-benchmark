import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--epochs', nargs='?', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of samples to load simultaneously')

    args = parser.parse_args()
    return args
