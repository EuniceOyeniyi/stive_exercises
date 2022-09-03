from statistics import mode
import torch

model = torch.nn.Linear(8,10)

img = torch.rand(10,10)

output = model(img)
print(output.shape)