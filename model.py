import torch.nn as nn

class Model(nn.Module):
    def __init__(self):

        self.conv0 = nn.Conv3d(1,32,(3,3,3))
        self.conv1 = nn.Conv3d(32,32,(4,4,4))
        self.conv2 = nn.Conv3d(32,32,(3,3,3))
        self.pool0 = nn.MaxPool3d((2,2,2),stride=(2,2,2))
        self.pool1 = nn.MaxPool3d((1,1,2),stride=(1,1,2))
        self.pool2 = nn.MaxPool3d((2,2,1),stride=(2,2,1))
        self.DropOut = nn.Dropout3d(p=0.2)
        self.activation = nn.RelU()
        self.bn = nn.BatchNorm3d(32)
        self.fc0 = nn.Linear(5 * 5 * 1, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        out = self.conv0(x)
        out = self.activation(out)
        out = self.pool0(out)
        out = self.bn(out)
        out = self.DropOut(out)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.pool1(out)
        out = self.bn(out)
        out = self.DropOut(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.pool2(out)
        out = self.bn(out)
        out = self.DropOut(out)
        out = out.view(-1, 5 * 5 * 1)
        out = self.fc0(out)
        out = self.activation(out)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.fc2(out)
        return out

model=Model()