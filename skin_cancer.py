# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:17:13 2019

@author: abdel
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.image as mpimg
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

data_dir = 'dataset'

train_on_gpu = torch.cuda.is_available() 
if not train_on_gpu :
    print("CUDA is not available: Training on cpu ...")
else:
    print("CUDA is available: Training on gpu ...")
# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                       transforms.RandomRotation(40),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=True)

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg16(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

print('classifier \n') 
model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(4096, 2048),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(2048, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.25),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
print('optimizer \n') 
# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device);
print('training beginning \n') 
epochs = 5
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        del inputs, labels, logps
        torch.cuda.empty_cache()
        running_loss += loss.item()
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                   
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    del inputs, labels, logps
                    torch.cuda.empty_cache()
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
            
torch.save(model, 'model_skin_cancerv2.pt')


model = torch.load('model_skin_cancerv2.pt')

# specify the image classes
classes = ['malignant', 'benign']
def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax  
# obtain one batch of test images
dataiter = iter(testloader)
images, labels = dataiter.next()


# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()
# get sample outputs
output = model(images)
# convert output probabilities to predicted class
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())
# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(10):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images.cpu()[idx])
    ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))      

    
ps = torch.exp(output)
#from PIL import Image
#def image_loader(data_dir):
#    image_transforms = transforms.Compose([transforms.Resize(256),
#                                      transforms.CenterCrop(224),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize([0.485, 0.456, 0.406],
#                                                           [0.229, 0.224, 0.225])])
#    image = Image.open(data_dir)
#    image = image_transforms(image).float()
##    image = Variable(image, requires_grad=True)
#    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
#    return image #assumes that you're using GPU
#
#def skin_cancer(image_dir):
#    image = image_loader(image_dir)
#    diagnosis = model(image.to(device))
#    diagnosis = torch.exp(diagnosis)
#    device1 = torch.device("cpu")
#    diagnosis = diagnosis.to(device1)
#    diagnosis = diagnosis.detach() 
#    return diagnosis.numpy()


plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)





