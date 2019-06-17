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
dataloader = dataloader()

is_train = True
data_image, data_loader_image = dataloader.load_data(is_train)

model = models.vgg16(pretrained=True)
print(model)

'''for parma in model.parameters():
    parma.requires_grad = False'''
model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))
model.to(device)

cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.ASGD(model.classifier.parameters())
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
            #print(X)
            y_pred = model(X)
            '''print(y_pred)'''
            _, pred = torch.max(y_pred.data, 1)
            '''print(_)
            print(pred)'''
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

torch.save(model.state_dict(), 'vgg16_catdog.pkl') # parameters'''