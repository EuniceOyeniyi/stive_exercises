from random import shuffle
from unittest import TestLoader
import torch 
from torchvision import datasets, transforms

def transformed_data():
    # Define a transform to normalize the d
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5),(0.5))])

    # Download and load the train and test data
    trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,batch_size = 64,shuffle= True)


    testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform) 
    testloader = torch.utils.data.DataLoader(testset,batch_size=64,shuffle=True)
 
    return trainloader,testloader

