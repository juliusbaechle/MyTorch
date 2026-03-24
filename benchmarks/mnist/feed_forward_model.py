import mytorch.nn as nn

class FeedForwardModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1 = nn.Linear(784, 196, device=device)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(196, 49, device=device)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(49, 24, device=device)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(24, 10, device=device)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        x = self.relu3(x)
        x = self.l4(x)
        return x