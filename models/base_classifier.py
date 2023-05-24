from torch import nn


class ConvBase(nn.Module):
    """
    Copied from: https://www.kaggle.com/code/shadabhussain/cifar-10-cnn-using-pytorch
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),

                                   nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),

                                   nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2),

                                   nn.Flatten(),
                                   nn.Linear(256 * 4 * 4, 1024),
                                   nn.ReLU(),
                                   nn.Linear(1024, 512),
                                   nn.ReLU(),
                                   nn.Linear(512, 10))

    def forward(self, x):
        return self.model(x)
