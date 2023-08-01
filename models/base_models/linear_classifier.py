from torch import nn


class LinearModel(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        out = self.model(x)
        return out
