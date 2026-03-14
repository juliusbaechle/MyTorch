import os
os.environ["SYCL_PI_LEVEL_ZERO_SYNC"] = "1"
#os.environ["SYCL_PI_TRACE"] = "2"

import faulthandler
faulthandler.enable()

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import mytorch
import mytorch.nn as nn
import mytorch.nn.functional as F

DEVICE = "gpu:0"



class MyModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1 = nn.Linear(784, 512, device=device)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(512, 256, device=device)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(256, 128, device=device)
        self.relu3 = nn.ReLU()
        self.l4 = nn.Linear(128, 10, device=device)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        x = self.relu2(x)
        x = self.l3(x)
        x = self.relu3(x)
        x = self.l4(x)
        return x

model = MyModel(DEVICE)
sgd = mytorch.SGD(model.parameters(), 0.001)
scheduler = mytorch.MultiStepLR(sgd, [6, 8], 0.1)



from mnist_dataset import MnistDatset
from data_loader import DataLoader

NUM_EPOCHS = 10
BATCH_SIZE = 500

train_dataset = MnistDatset("benchmarks\\mnist\\dataset", True)
train_loader = DataLoader(train_dataset, BATCH_SIZE, device=DEVICE)



import tqdm

for epoch in range(NUM_EPOCHS):
    pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")

    for X, y in pbar:
        logits = model.forward(X.reshape(-1, 784))
        loss = F.cross_entropy(logits, F.onehot(y, 10))
        pbar.set_postfix(loss="%.5f" % loss.item())

        sgd.zero_grad()
        loss.backward()
        sgd.step()
    
    scheduler.step()



test_dataset = MnistDatset("benchmarks\\mnist\\dataset", False)
test_loader = DataLoader(test_dataset, BATCH_SIZE, device=DEVICE)

num_correct = 0
for X, y in tqdm.tqdm(test_loader):
    with mytorch.no_grad():
        logits = model.forward(X.reshape(-1, 784))
        prediction = logits.argmax(dim=1)
        prediction = prediction == y
        num_correct += prediction.sum()

print(100 * num_correct / len(test_dataset))