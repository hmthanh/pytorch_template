import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim


class MyModel(nn.Module):
    def __init__(self, input_dim, n_hidden, output_dim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=n_hidden)
        self.fc2 = nn.Linear(in_features=n_hidden, out_features=output_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out


data_size = 100
hidden_layer = 32
label_size = 10
batch_size = 100
model = MyModel(input_dim=data_size, n_hidden=hidden_layer, output_dim=label_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(100):
    for batch in range(batch_size):
        x = torch.rand(data_size)
        y = torch.rand(label_size)

        model.train()
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            print("Epoch {} Batch {} Loss {}".format(epoch, batch, loss.item()))
