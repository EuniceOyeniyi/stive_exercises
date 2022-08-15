import torch as T
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from cnn_model import CNN

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

base_path = '../../Datasets'

transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,))
])

batch_size = 64

trainset = FashionMNIST(base_path + '/FMNIST_data', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = FashionMNIST(base_path + '/FMNIST_data', download=True, train=False, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

model = CNN()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3) # 1 * 10^-3 -> 0.001
criterion = nn.NLLLoss()

epochs = 5
train_losses = []
test_losses = []
accuracies = []

for epoch in range(epochs):

    tr_running_loss = 0
    te_running_loss = 0
    te_running_acc = 0

    for tr_imgs, tr_lbls in iter(trainloader):

        tr_imgs = tr_imgs.to(device)
        tr_lbls = tr_lbls.to(device)

        optimizer.zero_grad()
        tr_log_probs = model(tr_imgs)
        tr_loss = criterion(tr_log_probs, tr_lbls)
        tr_loss.backward()
        optimizer.step()

        tr_running_loss += tr_loss.item()

    tr_epoch_loss = tr_running_loss/len(trainloader)

    model.eval()
    with T.no_grad():
        for te_imgs, te_lbls in iter(testloader):

            te_imgs = te_imgs.to(device)
            te_lbls = te_lbls.to(device)

            te_log_probs = model(te_imgs)
            te_loss = criterion(te_log_probs, te_lbls)

            te_running_loss += te_loss.item()

            classes = T.exp(te_log_probs).argmax(dim=1)
            accuracy = sum(classes == te_lbls)/len(classes)

            te_running_acc =+ accuracy.item()
    model.train()
    
    te_epoch_loss = te_running_loss/len(testloader)
    epoch_accuracy = te_running_acc/len(testloader)

    train_losses.append(tr_epoch_loss)
    test_losses.append(te_epoch_loss)
    accuracies.append(epoch_accuracy)

    print(f'Epoch: {epoch} | Train loss: {tr_epoch_loss} | Test loss: {te_epoch_loss} | Accuracy: {epoch_accuracy}')

