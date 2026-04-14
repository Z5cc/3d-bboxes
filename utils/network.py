import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.pool1 = nn.AvgPool2d((2,2),stride=(2,2))
        self.pool2 = nn.AvgPool2d((2,2),stride=(2,2))
        self.pool3 = nn.AvgPool2d((2,2),stride=(2,2))
        self.pool4 = nn.AvgPool2d((2,2),stride=(2,2))
        self.pool5 = nn.AvgPool2d((2,2),stride=(2,2))
        self.conv1 = nn.Conv2d(4,8,(3,3),padding='same')
        self.conv2 = nn.Conv2d(8,16,(3,3),padding='same')
        self.conv3 = nn.Conv2d(16,32,(3,3),padding='same')
        self.conv4 = nn.Conv2d(32,32,(3,3),padding='same')
        self.conv5 = nn.Conv2d(32,32,(3,3),padding='same')
        self.conv6 = nn.Conv2d(32,32,(3,3),padding='same')
        self.conv7 = nn.Conv2d(32,32,(3,3),padding='same')
        self.conv8 = nn.Conv2d(32,32,(3,3),padding='same')
        self.conv9 = nn.Conv2d(32,32,(3,3),padding='same')
        self.conv10 = nn.Conv2d(32,32,(3,3),padding='same')
        self.lin1 = nn.Linear(8*8*32, 512)
        self.lin2 = nn.Linear(512,128)
        self.lin3 = nn.Linear(128,9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool3(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool4(x)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pool5(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x
