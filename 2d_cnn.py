import torch.nn as nn
from train import gen_2d_data
from sklearn import preprocessing
import torch
import torch.utils.data as tchdata

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np


class My2DConv(nn.Module):
    def __init__(self):
        super(My2DConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool2 = nn.MaxPool2d((2, 1))
        self.fc1 = nn.Linear(128 * 4 * 12, 300)
        self.fc2 = nn.Linear(300, 21)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 128 * 4 * 12)     
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


def main():
    n_samples = 20
    step = 2
    target = [1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]

    train_data, train_labels = gen_2d_data(target, n_samples, step, is_train=True)
    test_data, test_labels = gen_2d_data(target, n_samples, step, is_train=False)

    # scaler = preprocessing.StandardScaler().fit(train_data)
    # train_data = scaler.transform(train_data)
    # test_data = scaler.transform(test_data)

    train_dataset = tchdata.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
    test_dataset = tchdata.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))

    train_loader = tchdata.DataLoader(train_dataset, batch_size=300, shuffle=True)
    test_loader = tchdata.DataLoader(test_dataset, batch_size=300, shuffle=True)


if __name__ == '__main__':
    main()