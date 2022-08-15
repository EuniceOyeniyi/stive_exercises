import torch as T
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64*6*6, 74)
        self.fc2 = nn.Linear(74, 128)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        # MNIST images are 1 x 28 x 28
        x = T.relu(self.conv1(x)) # (D - k + 2P)//s + 1 => (28 - 3 + 2) // 2 + 1 = 14
        # x = self.mp(x) # (D - k + 2P)//s + 1 => (14 - 2 + 2*0) // 2 + 1 = 7
        x = T.relu(self.conv2(x)) # (D - k + 2P)//s + 1 => (7 - 3) // 1 + 1 = 5
        # x = self.mp(x) # (D - k + 2P)//s + 1 => (5 - 2) // 2 + 1 = 2
        x = T.relu(self.conv3(x)) # (D - k + 2P)//s + 1 => (2 - 3 + 2) // 2 + 1 = 1

        print(x.shape)
        x = x.view(x.shape[0], -1)

        x = T.relu(self.fc1(x))
        x = T.relu(self.fc2(x))
        return T.log_softmax(self.out(x), dim=1)

############## Model testing #######################
model = CNN()

imgs = T.rand((8, 1, 28, 28)) # Expected shape by a CNN: (batch_size, n_channels, width, heigth)

log_preds = model.forward(imgs)

print(log_preds)