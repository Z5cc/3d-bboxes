import torch
import torch.nn as nn
import torch.nn.functional as F

from Constants import K



class Head(nn.Module):
    def __init__(self, c_last_layer):
        super().__init__()
        self.lin1 = nn.Linear(8 * 8 * c_last_layer, 32 * c_last_layer)
        self.lin2 = nn.Linear(32 * c_last_layer, 8 * c_last_layer)
        self.lin3 = nn.Linear(8 * c_last_layer, 3)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x



class Model(nn.Module):

    def __init__(self, k=K):
        super().__init__()
        c1 = 1*k
        c2 = 2*k
        c3 = 4*k

        self.conv1 = nn.Conv2d(4, c1, 3, padding="same")
        self.conv2 = nn.Conv2d(c1, c1, 3, padding="same")

        self.conv3 = nn.Conv2d(c1, c2, 3, padding="same")
        self.conv4 = nn.Conv2d(c2, c2, 3, padding="same")

        self.conv5 = nn.Conv2d(c2, c3, 3, padding="same")
        self.conv6 = nn.Conv2d(c3, c3, 3, padding="same")

        self.conv7 = nn.Conv2d(c3, c3, 3, padding="same")
        self.conv8 = nn.Conv2d(c3, c3, 3, padding="same")

        self.conv9 = nn.Conv2d(c3, c3, 3, padding="same")
        self.conv10 = nn.Conv2d(c3, c3, 3, padding="same")

        self.pool = nn.MaxPool2d(2, stride=2)

        self.head_shift = Head(c3)
        self.head_scale = Head(c3)
        self.head_rotate = Head(c3)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(x)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.pool(x)

        x = torch.flatten(x, start_dim=1)

        x_shift = self.head_shift(x) # [N,3]
        x_scale = self.head_scale(x) # [N,3]
        x_rotate = self.head_rotate(x) # [N,3]
        x = torch.cat([x_shift, x_scale, x_rotate],dim=1) # [N,9]

        bb = self.create_bb(x) # [N,8,3]
        return bb


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
