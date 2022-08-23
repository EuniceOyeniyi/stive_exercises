import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from datahandler import data
from mnist_model import CNNmnist

# https://discuss.pytorch.org/t/view-classify-in-module-helper/30279/6

def view_classify(img, ps):

    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.show()


#loding the model and the datasets 
model = CNNmnist()
state_dict = torch.load('model_ 0.99.pth')
model.load_state_dict(state_dict)
trainloader,testloader = data()

#Checking if the prediction is correct

images, labels = next(iter(testloader))
# images = images.view(images.shape[0], -1)
# with torch.no_grad():
#     output =model.forward(images[0,:])
# ps = torch.argmax(output, dim=1)
print(images.shape[0])