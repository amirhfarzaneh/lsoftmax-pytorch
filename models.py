from torch import nn
from lsoftmax import LSoftmaxLinear


class MNISTNet(nn.Module):
    def __init__(self, margin, device):
        super(MNISTNet, self).__init__()
        self.margin = margin
        self.device = device

        self.conv_0 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 64, 3),
            nn.PReLU(),
            nn.BatchNorm2d(64)
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(576, 256),
            nn.BatchNorm1d(256)
        )
        self.lsoftmax_linear = LSoftmaxLinear(
            input_features=256, output_features=10, margin=margin, device=self.device)
        self.reset_parameters()

    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()

    def forward(self, x, target=None):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(-1, 576)
        x = self.fc(x)
        logit = self.lsoftmax_linear(input=x, target=target)
        return logit


class MNISTFIG2Net(nn.Module):
    def __init__(self, margin, device):
        super(MNISTFIG2Net, self).__init__()
        self.margin = margin
        self.device = device

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, padding=2),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 5, padding=2),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2),
            nn.PReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 5, padding=2),
            nn.PReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(1152, 2),
            nn.BatchNorm1d(2)
        )
        self.lsoftmax_linear = LSoftmaxLinear(
            input_features=2, output_features=10, margin=margin, device=self.device)
        self.reset_parameters()

    def reset_parameters(self):
        self.lsoftmax_linear.reset_parameters()

    def forward(self, x, target=None):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = x.view(-1, 1152)
        x = self.fc(x)
        logit = self.lsoftmax_linear(input=x, target=target)
        return logit, x

