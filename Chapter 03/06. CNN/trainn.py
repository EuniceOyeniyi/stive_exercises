import torch as T
import matplotlib.pyplot as plt
from torch import optim,nn
import torch.nn.functional as F
from mnist_model import CNNmnist
from datahandler import data

device = T.device('cuda' if T.cuda.is_available() else 'cpu')
#Getting the data 
trainloader,testloader = data()
#Defining the loss function 

#Defining the optimizer 
model = CNNmnist()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.NLLLoss()

epochs = 15
train_losses = []
test_losses = []
accuracies = []
best_accuracy= 0.85

for e in range(epochs):
    print(f"Epoch: {e+1}/{epochs}")
    tr_running_loss = 0
    te_running_loss = 0
    te_running_acc = 0
    
    

    for i, (images, labels) in enumerate(iter(trainloader)):
        
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        output = model.forward(images)   
        loss = criterion(output, labels) 
        loss.backward()                  
        optimizer.step()                  
        
        tr_running_loss+=loss.item()
    training_loss = tr_running_loss/ len(trainloader)
    train_losses.append(training_loss)
        
        

    
    model.eval()
    correct_pred = 0
    total_size = 0
    running_loss_test = 0
    
    with T.no_grad():
        for i, (tst_images, tst_labels) in enumerate(iter(testloader)):
            
            tst_images = tst_images.to(device)
            tst_labels = tst_labels.to(device)

         
            #Get prediction by running images throught the network
            output2 = model(tst_images)
            prediction = output2.argmax(dim=1)

            #getting accuracy
            correct_pred+=(prediction==tst_labels).sum()
            total_size += tst_labels.size(0)
            

            #getting the loss value
            loss2 = criterion(output2,tst_labels)
            running_loss_test+=(loss2.item())  

        testing_loss = running_loss_test/len(testloader)
        test_losses.append(testing_loss)


        accuracy = correct_pred / total_size
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            T.save(model.state_dict(), f'model_{best_accuracy: .2f}.pth')


    model.train()
    print(f'epoch: {e} | loss: {loss2.item()} | accuracy: {accuracy.item()}')

plt.figure(figsize=(15,8))
plt.plot(train_losses, label='Train loss')
plt.plot(test_losses, label='Test loss')
plt.plot(accuracies, label='Accuracy')
plt.legend()
plt.show()

