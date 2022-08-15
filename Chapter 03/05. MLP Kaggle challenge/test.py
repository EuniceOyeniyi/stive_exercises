import torch
import matplotlib.pyplot as plt
import numpy as np

from datahandler import transformed_data
from model import model
#  loading model
state_dict = torch.load('models_ 0.90.pth')
model.load_state_dict(state_dict)
trainloader,testloader = transformed_data()
dataiter = iter(testloader)
images, labels = dataiter.next()
images = images.view(images.shape[0],-1) #flatten the image
probs = model(images)

top_p, top_class = probs.topk(1, dim=1)
# # Look at the most likely classes for the first 10 examples

equals = top_class == labels.view(*top_class.shape)
# equals = top_class == labels
misclassified = [index for index,value in enumerate(equals) if value.item() is False]
accuracy = torch.mean(equals.type(torch.FloatTensor))
print(f'Accuracy: {accuracy.item()*100 }% ') 