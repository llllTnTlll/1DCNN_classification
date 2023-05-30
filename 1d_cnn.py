import torch
import torch.nn as nn
from datetime import datetime
import numpy as np
import torch.utils.data as tchdata
from sklearn import preprocessing
import matplotlib.pyplot as plt
from train import AccMectric, gen_seq_data, train, validate, read_data
from torch.optim.lr_scheduler import ReduceLROnPlateau


class oned_CNN(nn.Module):
    def __init__(self, *, N_Features, N_ClassesOut, Conv1_NF=64, Conv2_NF=256, Conv3_NF=128, FC_DropP=0.3, n_samples, hidden=32):
        super(oned_CNN, self).__init__()
        self.sequence_length = n_samples
        self.N_Features = N_Features
        self.hiden_size = hidden
        self.sequence_length = n_samples
        self.N_ClassesOut = N_ClassesOut
        self.Conv1_NF = Conv1_NF
        self.Conv2_NF = Conv2_NF
        self.Conv3_NF = Conv3_NF

        self.C1 = nn.Conv1d(self.N_Features, self.Conv1_NF, 3)
        self.C2 = nn.Conv1d(self.Conv1_NF, self.Conv2_NF, 3)
        self.C3 = nn.Conv1d(self.Conv2_NF, self.Conv3_NF, 3)
        self.BN1 = nn.BatchNorm1d(self.Conv1_NF)
        self.BN2 = nn.BatchNorm1d(self.Conv2_NF)
        self.BN3 = nn.BatchNorm1d(self.Conv3_NF)
        self.relu = nn.ReLU()
        self.ConvDrop = nn.Dropout(FC_DropP)
        self.p1 = nn.MaxPool1d(kernel_size=2, padding=1)
        # self.FC = nn.Linear(self.Conv3_NF, self.N_ClassesOut)
        self.FC = nn.Linear(96, self.hiden_size)
        self.b1 = torch.nn.BatchNorm1d(self.hiden_size)
        self.sm = torch.nn.Linear(self.hiden_size, self.N_ClassesOut)

    def forward(self, x):
        seq_data = x.chunk(self.sequence_length, dim=1)
        feature_map = torch.cat([data.unsqueeze(1) for data in seq_data], dim=1)    # (B, T, F)
        feature_map = feature_map.permute(0, 2, 1)
        # x2 = self.C1(seq_data[0])
        x2 = self.C1(feature_map)
        x2 = self.ConvDrop(self.relu(self.BN1(x2)))
        # x2 = self.SE1(x2)
        x2 = self.p1(x2)
        x2 = self.C2(x2)
        x2 = self.ConvDrop(self.relu(self.BN2(x2)))
        # x2 = self.SE2(x2)
        x2 = self.p1(x2)
        x2 = self.C3(x2)
        x2 = self.ConvDrop(self.relu(self.BN3(x2)))
        x2 = x2.view(-1, self.Conv3_NF * 3)
        # x2 = torch.mean(x2, 1)  # Global average pooling --> [B, Conv3_NF]
        x = self.FC(x2)
        x = self.b1(x)
        x_out = self.sm(x)
        return x_out


def main():
    n_samples = 20

    # target = [1, 2, 6, 7, 8]              # case1
    # target = [3, 4, 5, 9, 10, 11, 12]     # case2
    target = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    # target = list(range(1, 22))           # total

    train_data, train_labels = gen_seq_data(target, n_samples, is_train=True)
    test_data, test_labels = gen_seq_data(target, n_samples, is_train=False)

    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)

    train_dataset = tchdata.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
    test_dataset = tchdata.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))

    train_loader = tchdata.DataLoader(train_dataset, batch_size=300, shuffle=True)
    test_loader = tchdata.DataLoader(test_dataset, batch_size=300, shuffle=True)

    model = oned_CNN(N_Features=52, N_ClassesOut=len(target), Conv1_NF=64, Conv2_NF=128, Conv3_NF=32, FC_DropP=0.3,
                     n_samples=n_samples, hidden=100)

    model.cuda()
    torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=0.01)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=15, verbose=True, min_lr=0.0003)
    # scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=True, min_lr=5e-5)

    train_accs, test_accs = [], []

    for i in range(60):
        train_acc = train(model, optimizer, train_loader)
        test_acc = validate(model, test_loader)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print('{}\tepoch = {}\ttrain accuracy: {:0.3f}\ttest accuracy: {:0.3f}'.format(datetime.now(), i, train_acc, test_acc))
        # scheduler.step(test_acc, epoch=i)

    # 绘制训练精度和测试精度
    print(np.mean(train_accs))
    print(np.mean(test_accs))

    epochs = range(1, i + 2)
    plt.plot(epochs, train_accs, '*-', label='Training accuracy', c="k")
    plt.plot(epochs, test_accs, '*-', label='Validating accuracy', c="r")
    plt.title('1d_cnn')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.yticks([i/100 for i in range(0, 81, 10)]+[i/100 for i in range(80, 101, 4)])
    for y in range(80,  101, 2):
        plt.axhline(y / 100, linestyle='--', color='gray', lw=0.5)
    plt.show()

    # torch.save(model, 'att_indrnn_FCN')


if __name__ == '__main__':
    main()

