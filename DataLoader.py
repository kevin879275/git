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

class dataloader():
    def load_data(self,is_training):
        if(is_training == True):
            path = './catdog'
            transform = transforms.Compose([transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

            datasets.ImageFolder(root=path)
            {x: print(x, end=", ") for x in ["train", "val"]}

            data_image = {x: datasets.ImageFolder(root=path + '/' + x, transform=transform) for x in ["train", "val"]}
            data_image
            data_image["train"]
            classes = data_image["train"].classes
            classes_index = data_image["train"].class_to_idx
            print(classes)
            print(classes_index)
            data_loader_image = {x: torch.utils.data.DataLoader(dataset=data_image[x], batch_size=6, shuffle=True)
                                 for x in ["train", "val"]}
            return data_image, data_loader_image
        else:
            path = './4_test_cats_dogs'
            transform = transforms.Compose([transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

            # Google 找幾張 cat, dog images 放到 Cat_dog_test/test 資料夾
            data_test_img = datasets.ImageFolder(root=path, transform=transform)
            test_number = len(data_test_img)
            data_loader_test_img = torch.utils.data.DataLoader(dataset=data_test_img, batch_size=test_number)
            image, _ = next(iter(data_loader_test_img))
            image.shape

            return test_number, data_loader_test_img

