import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

from model.model import MyModel
from configs.parse_args import parse_args


def main(config, device):
    print(config)

    inputs = torch.randn(config.batch_size, config.input_dim)

    model = MyModel(config.input_dim, config.n_hidden, config.output_dim)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()

    output = model(inputs)
    print("output", output)
    print(output.size())


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.gpu)
    print("args", args)
    with open(args.configs) as f:
        config = yaml.safe_load(f)

    for k, v in vars(args).items():
        config[k] = v

    config = EasyDict(config)
    main(config, device)
