import torch
import torchvision
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets,transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt
import time
from DataLoader import dataloader


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(device)

is_train = False;
dataloader = dataloader()

data_image, data_loader_test_img = dataloader.load_data(is_train)

vgg16 = models.vgg16(pretrained=True)

vgg16.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))

vgg16.load_state_dict(torch.load('vgg16_catdog.pkl'))
vgg16.to(device)

image, _ = next(iter(data_loader_test_img))
image.shape
images = Variable(image).to(device)
y_pred = vgg16(images)
_, pred = torch.max(y_pred.data, 1)
print(pred)
imgArray = torchvision.utils.make_grid(image)
imgArray = imgArray.numpy()
imgArray = imgArray*0.5+0.5
print(imgArray.shape, imgArray.min(), imgArray.max())


lst = list(pred.cpu().numpy())
for i in (lst):
    if i == 0:
        print("cat", end=", ")
    elif i == 1:
        print("dog", end=", ")
imgArray1 = np.zeros((228, 1358, 3))
imgArray1[:,:,0] = imgArray[0, :, :]
imgArray1[:,:,1] = imgArray[1, :, :]
imgArray1[:,:,2] = imgArray[2, :, :]
imgArray1.shape
plt.figure(figsize=(18, 6))
plt.imshow(imgArray1)
plt.show()