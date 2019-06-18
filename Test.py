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
import tkinter as tk

dataloader = dataloader()
def hit_me():
    global on_hit

    var.set('asdasdasd')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    is_train = False;

    test_number, data_loader_test_img = dataloader.load_data(is_train)

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
    imgArray = imgArray * 0.5 + 0.5
    print(imgArray.shape, imgArray.min(), imgArray.max())

    lst = list(pred.cpu().numpy())
    for i in (lst):
        if i == 0:
            print("cat", end=", ")
        elif i == 1:
            print("dog", end=", ")
    if (test_number < 9):
        imgArray1 = np.zeros((228, 224 * test_number + (2 * test_number + 2), 3))
    else:
        imgArray1 = np.zeros((228 + (226 * (test_number // 8)), 1810, 3))
    imgArray1[:, :, 0] = imgArray[0, :, :]
    imgArray1[:, :, 1] = imgArray[1, :, :]
    imgArray1[:, :, 2] = imgArray[2, :, :]
    imgArray1.shape
    plt.figure(figsize=(18, 6))
    plt.imshow(imgArray1)
    plt.show()

window = tk.Tk()
window.title('my window')
window.geometry('200x100')

var = tk.StringVar()    # 这时文字变量储存器
l = tk.Label(window,
    textvariable=var,   # 使用 textvariable 替换 text, 因为这个可以变化
    bg='green', font=('Arial', 12), width=15, height=2)
l.pack()

b = tk.Button(window,
    text='hit me',      # 显示在按钮上的文字
    width=15, height=2,
    command=hit_me)     # 点击按钮式执行的命令
b.pack()    # 按钮位置

window.mainloop()



