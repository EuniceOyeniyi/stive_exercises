
import torch 
import torch.nn as nn
import torch.nn.functional as F

class CNNmnist(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3,padding='same')
        self.conv2 = nn.Conv2d(32,64,3,padding='same')
        self.max_pool = nn.MaxPool2d(2,2)


        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128,300)
        self.out = nn.Linear(300,10)

    def forward(self,x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
    
        

        # print(x.shape)
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))    

        return torch.log_softmax(self.out(x),dim=1)

# model = CNNmnist()
# fake = torch.rand((8, 1, 28, 28))
# model.forward(fake)
