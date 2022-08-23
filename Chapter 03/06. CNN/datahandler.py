import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms


def data():
    # Define a transform to normalize the data (Preprocessing)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5)) ])

    # Download and load the training data
    trainset    = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    # Download and load the test data
    testset    = datasets.MNIST('MNIST_data/', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return trainloader,testloader