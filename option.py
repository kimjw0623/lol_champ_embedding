import argparse

parser = argparse.ArgumentParser()

# Settings
parser.add_argument('--region', type=str, default='KR')

parser.add_argument('--tier', type=str, default='master')

parser.add_argument('--get_data', type=bool, default=False)

parser.add_argument('--is_train', type=bool, default=False)

# Hyperparameters
parser.add_argument('--embed_num', type=int, default=15)

parser.add_argument('--tot_epoch', type=int, default=150)

parser.add_argument('--batch_size', type=int, default=64)

args = parser.parse_args()
