from torch import nn


class ToyModel(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_dim, input_dim*10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(input_dim*10, input_dim*10)
        self.fc3 = nn.Linear(input_dim*10, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
