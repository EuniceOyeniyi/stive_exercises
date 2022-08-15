import torch
import matplotlib.pyplot as plt
from torch import optim,nn
import torch.nn.functional as F
from model import model
from datahandler import transformed_data

#Getting the data 
trainloader,testloader = transformed_data()
#Defining the loss function 
criterion = nn.CrossEntropyLoss()
#Defining the optimizer 
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
print_every = 60
train_loss = []
test_loss = []
accuracies = []
best_accuracy = 0.85
for e in range(epochs):
    print(f"Epoch: {e+1}/{epochs}")
    running_loss_train = 0
    
    

    for i, (images, labels) in enumerate(iter(trainloader)):

        # Flatten MNIST images into a 784 long vector
        images.resize_(images.size()[0], 784)
        
        optimizer.zero_grad()
        
        output = model.forward(images)   # 1) Forward pass
        loss = criterion(output, labels) # 2) Compute loss
        loss.backward()                  # 3) Backward pass
        optimizer.step()                  # 4) Update model
        
        running_loss_train+=(loss.item())
    training_loss = running_loss_train/ len(trainloader)
    train_loss.append(training_loss)
        
        
        # if i % print_every == 0:
        #     print(f"\tIteration: {i}\t Loss: {training_loss/print_every:.4f}")
        #     training_loss = 0
    
    model.eval()
    correct_pred = 0
    total_size = 0
    running_loss_test = 0
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(iter(testloader)):

            #reshaping images
            images.resize_(images.size()[0], 784)

            #Prediction
            output2 = model(images)
            prediction = F.softmax(output2, dim=1).argmax(dim=1)

            #getting accuracy
            correct_pred+=(prediction==labels).sum()
            total_size += labels.size(0)
            

            #getting the loss value
            loss2 = criterion(output2,labels)
            running_loss_test+=(loss2.item())  

        testing_loss = running_loss_test/len(testloader)
        test_loss.append(testing_loss)


        accuracy = correct_pred / total_size
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), f'models_{accuracy: .2f}.pth')

    model.train()
    print(f'epoch: {e} | test_loss: {loss2.item()} | accuracy: {accuracy.item()}')

plt.figure(figsize=(15,8))
plt.plot(train_loss, label='Train loss')
plt.plot(test_loss, label='Test loss')
plt.plot(accuracies, label='Accuracy')
plt.legend()
plt.show()