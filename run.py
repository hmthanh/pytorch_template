import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


class MyModel(nn.Module):
    def __init__(self, data_dim, n_hidden, output_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(data_dim, n_hidden)
        self.fc2 = nn.Linear(n_hidden, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out


input_dim = 100
hidden_layers = 128
output_dim = 10
model = MyModel(input_dim, hidden_layers, output_dim)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

batch_size = 32

for epoch in range(100):
    data = torch.rand(batch_size, input_dim)
    labels = torch.rand(batch_size, output_dim)

    optimizer.zero_grad()
    predict = model(data)
    loss = criterion(predict, labels)
    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('Epoch: {}, Loss: {}'.format(epoch + 1, loss.item()))
