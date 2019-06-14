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


use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(device)

IMG_SIZE = 80
batch_size = 8

path = './catdog'

transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

datasets.ImageFolder(root = path)
{x: print(x, end=", ") for x in ["train", "val"]}

data_image = {x:datasets.ImageFolder(root =path+'/'+x, transform = transform) for x in ["train", "val"]}
data_image
data_image["train"]
classes = data_image["train"].classes
classes_index = data_image["train"].class_to_idx
print(classes)
print(classes_index)
data_loader_image = {x:torch.utils.data.DataLoader(dataset=data_image[x], batch_size =20, shuffle = True)
                     for x in ["train", "val"]}
data_loader_image["train"]
it = iter(data_loader_image["train"])
first_batch_x, first_batch_y = next(it)
second_batch_x, second_batch_y = next(it)
print(first_batch_x.shape, first_batch_y.shape)
img = torchvision.utils.make_grid(first_batch_x)
print(img.shape)
imgArray = img.numpy()
print(imgArray.shape, imgArray.min(), imgArray.max())
imgArray = imgArray*0.5+0.5
print(imgArray.shape, imgArray.min(), imgArray.max())
imgArray1 = np.zeros((680, 1810, 3))
imgArray1[:,:,0] = imgArray[0, :, :]
imgArray1[:,:,1] = imgArray[1, :, :]
imgArray1[:,:,2] = imgArray[2, :, :]
imgArray1.shape
plt.figure(figsize=(18, 6))
plt.imshow(imgArray1)
plt.show()
print([classes[i] for i in first_batch_y])
model = models.vgg16(pretrained=True)
print(model)
for parma in model.parameters():
    parma.requires_grad = False
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))
model.to(device)

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters())
n_epochs = 10
for epoch in range(n_epochs):
    since = time.time()
    print("Epoch{}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for param in ["train", "val"]:
        if param == "train":
            model.train = True
        else:
            model.train = False

        running_loss = 0.0
        running_correct = 0
        batch = 0
        for data in data_loader_image[param]:
            batch += 1
            X, y = data
            X = Variable(X).to(device)
            y = Variable(y).to(device)

            optimizer.zero_grad()
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)

            loss = cost(y_pred, y)
            if param == "train":
                loss.backward()
                optimizer.step()
            running_loss += float(loss.data)
            running_correct += torch.sum(pred == y.data)
            if batch % 500 == 0 and param == "train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                    batch, running_loss / (4 * batch), 100 * running_correct / (4 * batch)))

        epoch_loss = running_loss / len(data_image[param])
        epoch_correct = 100 * running_correct / len(data_image[param])

        print("{}  Loss:{:.4f},  Correct{:.4f}".format(param, epoch_loss, epoch_correct))
    now_time = time.time() - since
    print("Training time is:{:.0f}m {:.0f}s".format(now_time // 60, now_time % 60))

torch.save(model.state_dict(), 'vgg16_catdog.pkl') # parameters

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(device)

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

transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

# Google 找幾張 cat, dog images 放到 Cat_dog_test/test 資料夾
data_test_img = datasets.ImageFolder(root="./4_test_cats_dogs", transform = transform)

data_loader_test_img = torch.utils.data.DataLoader(dataset=data_test_img, batch_size = 6)
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