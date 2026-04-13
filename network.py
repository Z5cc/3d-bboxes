import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool1 = nn.MaxPool2d((2,2),stride=(2,2))
        self.pool2 = nn.MaxPool2d((2,2),stride=(2,2))
        self.pool3 = nn.MaxPool2d((2,2),stride=(2,2))
        self.pool4 = nn.MaxPool2d((2,2),stride=(2,2))
        self.pool5 = nn.MaxPool2d((2,2),stride=(2,2))
        self.pool6 = nn.MaxPool2d((2,2),stride=(2,2))
        self.conv1 = nn.Conv2d(4,8,(3,3),padding='same')
        self.conv2 = nn.Conv2d(8,16,(3,3),padding='same')
        self.conv3 = nn.Conv2d(16,16,(3,3),padding='same')
        self.conv4 = nn.Conv2d(16,16,(3,3),padding='same')
        self.conv5 = nn.Conv2d(16,16,(3,3),padding='same')
        self.conv6 = nn.Conv2d(16,16,(3,3),padding='same')
        self.lin1 = nn.Linear(4*4*16, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = F.relu(self.conv5(x))
        x = self.pool5(x)
        x = F.relu(self.conv6(x))
        x = self.pool6(x)
        x = torch.flatten(x)
        x = self.lin1(x)
        return x
