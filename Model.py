import torch
import torch.nn as nn
import torch.nn.functional as F

from Constants import K




class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) # every conv layer is followed by BN, and BN already learns a 'bias', so not another 'bias' needed
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        # if stride>1: also the channels of the skip connections have to be reduced by 1x1 convolution with same stride.       resnet downsamples with stride=2. VGG used max_pooling.
        if stride>1:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False) 
        else:
            self.skip_connection = nn.Identity()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out =        self.bn2(self.conv2(out))
        x = self.skip_connection(x)
        out += x
        return F.relu(out)


# class ResModel(nn.Model):
    # def __init__(self):
    #     super().__init__()
    #     self.res1 = ResidualUnit()
    #     self.res2 = ResidualUnit()
    #     self.res3 = ResidualUnit()
    #     self.pool = nn.MaxPool2d(2, stride=2)
    






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


        self.conv11 = nn.Conv2d(c3, c3, 3, padding="same")
        self.conv12 = nn.Conv2d(c3, c3, 3, padding="same")
        self.conv13 = nn.Conv2d(c3, c3, 3, padding="same")
        self.conv14 = nn.Conv2d(c3, c3, 3, padding="same")
        self.conv15 = nn.Conv2d(c3, c3, 3, padding="same")

        self.pool = nn.MaxPool2d(2, stride=2)


        self.lin1 = nn.Linear(8 * 8 * c3, 512*40)
        self.lin2 = nn.Linear(512*40, 128*40)
        self.lin3 = nn.Linear(128*40, 9)



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        # x = F.relu(self.conv11(x))
        x = self.pool(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        # x = F.relu(self.conv12(x))
        # x = F.relu(self.conv13(x))
        x = self.pool(x)
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        # x = F.relu(self.conv14(x))
        # x = F.relu(self.conv15(x))
        x = self.pool(x)
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
