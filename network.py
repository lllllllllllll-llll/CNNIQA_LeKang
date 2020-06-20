import torch
from torch import nn
import torch.nn.functional as F



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=7)
        self.fc1 = nn.Linear(100, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 1)

    def forward(self, input):
        x = input.view(-1, input.size(-3), input.size(-2), input.size(-1))
        # print('x', x.size())#[128, 1, 32, 32]

        x = self.conv(x)
        # print('x', x.size())#[128, 50, 26, 26]

        x1 = F.max_pool2d(x, (x.size(-2), x.size(-1)))
        # print('x1', x1.size())#[128, 50, 1, 1]
        x2 = -F.max_pool2d(-x, (x.size(-2), x.size(-1)))
        # print('x2', x2.size())#[128, 50, 1, 1]

        h = torch.cat((x1, x2), 1)
        # print('h', h.size())#[128, 100, 1, 1]
        h = h.squeeze(3).squeeze(2)
        # print('h', h.size())#[128, 100]

        h = F.relu(self.fc1(h))
        h = F.dropout(h)
        h = F.relu(self.fc2(h))

        q = self.fc3(h)

        return q


