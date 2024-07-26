import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MyDataset(Dataset):
    def __init__(self, batch_size, input_dim, output_dim):
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.inputs = np.array(np.random.uniform(0, 10, size=(batch_size, input_dim)), dtype=np.float32)
        self.labels = torch.randn(output_dim)
        self.labels = torch.randint(0, output_dim, (batch_size,))

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


if __name__ == '__main__':
    from configs.parse_args import parse_args

    configs = parse_args()
    dataset = MyDataset(batch_size=configs.batch_size, input_dim=configs.input_dim, output_dim=configs.output_dim)
    data_loader = DataLoader(dataset, batch_size=configs.batch_size, shuffle=True)
    for batch_idx, (data, labels) in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}")
        print(f"Data: {data}")
        print(f"Labels: {labels}")
        print("=" * 20)
    print("dataset", dataset[0])
