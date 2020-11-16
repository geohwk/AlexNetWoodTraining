
# Imports
from torchvision.datasets import ImageFolder 
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Normalize
from torchvision import transforms
import torch
import os
import torchvision
import torch.nn as nn                                                                           # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim                                                                     # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F                                                                 # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
)                                                                                               # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets                                                         # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms                                                     # Transformations we can perform on our dataset

root_dir = os.path.join(os.path.dirname(__file__), 'PreTrained_Format_Images')

#Transform for our dataset to be able to be fed into alexNet
trainTransforms = transforms.Compose([Resize((224, 224)),ToTensor(),Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameters
num_classes = 3
learning_rate = 0.0001
batch_size = 4
num_epochs = 5

# Simple Identity class that let's input pass without changes
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

model = torchvision.models.alexnet(pretrained = True)                                           # Load pretrain model & modify it

model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 3)                       #Edits last layer to only output 2 classes, wood and plastic

model.to(device)

# Load Data
train_dataset = ImageFolder(root = os.path.join(root_dir, 'train'),transform= trainTransforms)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImageFolder(root = os.path.join(root_dir, 'val'),transform= trainTransforms)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    #if loader.dataset.train:
    #    print("Checking accuracy on training data")
    #else:
    print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()

check_accuracy(test_loader, model)


single_Dataset = ImageFolder(root = os.path.join(root_dir, 'singleimage'),transform= trainTransforms)

single_loader = DataLoader(dataset=single_Dataset, batch_size=1, shuffle=True)


for x, y in single_loader:
    x = x.to(device=device)
    scores = model(x)
    _, predictions = scores.max(1)
    if predictions == 0:
        print("Random Image")
    elif predictions == 1:
        print("Plastic")
    elif predictions == 2:
        print("Wood")

