import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np

from configs.parse_args import parse_args
from dataset.dataset import MyDataset
from model.model import MyModel


def main(config, device):
    print(config)

    # Loading model
    low = 0.0
    high = 5.0
    uniform_data = np.random.uniform(low, high, size=(config.batch_size, config.input_dim))
    # data = torch.from_numpy(np.array(uniform_data, dtype=np.float32)).to(device)
    # labels = torch.randint(0, config.output_dim, (config.batch_size,)).to(device)

    # Create model
    model = MyModel(config.input_dim, config.n_hidden, config.output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.NLLLoss()

    # Dataloader
    dataset = MyDataset(batch_size=config.batch_size, input_dim=config.input_dim, output_dim=config.output_dim)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Training
    model.train()
    for epoch in range(config.epoch):
        for batch_idx, (data, labels) in enumerate(data_loader):
            device_data = data.to(device)
            device_labels = labels.to(device)

            optimizer.zero_grad()

            output = model(device_data)
            loss = criterion(output, device_labels)
            loss.backward()

            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print('Epoch: {}, Loss: {}'.format(epoch + 1, loss.item()))

    # Save model
    model = model.to(torch.device('cpu'))
    torch.save(model.state_dict(), config.model_path)


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
