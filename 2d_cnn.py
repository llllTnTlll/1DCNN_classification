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

import torch
from torch import nn
from torch.nn import functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # 池化层
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1))

        # 全连接层
        self.fc1 = nn.Linear(128 * 5 * 26, 300)
        self.fc2 = nn.Linear(300, 21)

    def forward(self, x):
        # 增加一个维度表示单通道
        x = x.unsqueeze(1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        # 展平
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # softmax
        output = F.log_softmax(x, dim=1)

        return output


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