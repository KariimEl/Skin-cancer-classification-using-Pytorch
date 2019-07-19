# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 01:40:33 2019

@author: abdel
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.image as mpimg
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    
from PIL import Image
def image_loader(data_dir):
    image_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    image = Image.open(data_dir)
    image = image_transforms(image).float()
#    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image #assumes that you're using GPU

def skin_cancer(image_dir,model):
    image = image_loader(image_dir)
    diagnosis = model(image.to(device))
    diagnosis = torch.exp(diagnosis)
    device1 = torch.device("cpu")
    diagnosis = diagnosis.to(device1)
    diagnosis = diagnosis.detach() 
    return diagnosis.numpy()

