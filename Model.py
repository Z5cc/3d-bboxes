import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = nn.MaxPool2d((2,2),stride=(2,2))
        self.pool2 = nn.MaxPool2d((2,2),stride=(2,2))
        self.pool3 = nn.MaxPool2d((2,2),stride=(2,2))
        self.pool4 = nn.MaxPool2d((2,2),stride=(2,2))
        self.pool5 = nn.MaxPool2d((2,2),stride=(2,2))
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
        x = self.create_bb(x)
        return x

    def create_bb(self, y): # [N,9]
        # 0. BASE OF BOUNDING BOX
        bb = torch.tensor([[-0.5,-0.5,-0.5],[-0.5,0.5,-0.5],[0.5,0.5,-0.5],[0.5,-0.5,-0.5],
                            [-0.5,-0.5,0.5],[-0.5,0.5,0.5],[0.5,0.5,0.5],[0.5,-0.5,0.5]],
                            dtype = torch.float, device=y.device)
        bb = bb[None,:,:] # [N,8,3]

        # 1. SCALE
        size = y[:,None, 3:6] # [N,1,3]
        size = F.softplus(size)
        bb = bb * size  # [N,8,3]

        # 2. ROTATE
        angles = (torch.tanh(y[:,6:9])) * (torch.pi/4) # [N,3]
        cx, cy, cz = torch.cos(angles[:,0]), torch.cos(angles[:,1]), torch.cos(angles[:,2]) # [N]
        sx, sy, sz = torch.sin(angles[:,0]), torch.sin(angles[:,1]), torch.sin(angles[:,2]) # [N]
        R = torch.zeros((y.shape[0], 3, 3),device=y.device) # [N,3,3]

        R[:,0,0] = cy * cz
        R[:,0,1] = cz * sx * sy - cx * sz
        R[:,0,2] = cx * cz * sy + sx * sz

        R[:,1,0] = cy * sz
        R[:,1,1] = cx * cz + sx * sy * sz
        R[:,1,2] = -cz * sx + cx * sy * sz

        R[:,2,0] = -sy
        R[:,2,1] = cy * sx
        R[:,2,2] = cx * cy

        bb = torch.matmul(bb, R.transpose(1,2)) # [N,8,3]=[N,8,3]*[N,3,3]

        # 3. SHIFT
        center = y[:,None,0:3] # [N,1,3]
        bb = bb + center

        return bb # [N,8,3]
