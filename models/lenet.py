'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        self.name = "LeNet"

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

if __name__ == "__main__":

    # net.named_parameters() 也是可迭代对象，既能调出网络的具体参数，也有名字信息
    # for name, parameters in lenet.named_parameters():
    #     print(name, ';', parameters.size())

    net = LeNet()
    print(net.name)
    summary(net, input_size=(3, 32, 32), device='cpu')