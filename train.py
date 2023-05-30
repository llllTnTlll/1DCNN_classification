import os
import numpy as np
import torch
import random


class AccMectric(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self._sum = 0
        self._count = 0
    
    def update(self, targets, outputs):
        pred = outputs.argmax(axis=1)
        self._sum += (pred == targets).sum()
        self._count += targets.shape[0]
        
    def get(self):
        return self._sum / self._count


def read_data(error=0, is_train=True):
    fi = os.path.join('data/dataset/',
        ('d0' if error < 10 else 'd') + str(error) + ('_te.dat' if is_train else '.dat'))
    with open(fi, 'r') as fr:
        data = fr.read()
    data = np.fromstring(data, dtype=np.float32, sep='   ')
    if fi == 'data/d00.dat':
        data = data.reshape(-1, 500).T
    else:
        data = data.reshape(-1, 52)
    if is_train:
        data = data[160:]
    return data, np.ones(data.shape[0], np.int64) * error


def gen_seq_data(target, n_samples, is_train):
    seq_data, seq_labels = [], []
    for i, t in enumerate(target):
        d, _ = read_data(t, is_train)   # 忽略标签
        data = []
        length = d.shape[0] - n_samples + 1
        for j in range(n_samples):
            data.append(d[j : j + length])
        data = np.hstack(data)
        seq_data.append(data)
        seq_labels.append(np.ones(data.shape[0], np.int64) * i)

        d = np.vstack(seq_data)
        l = np.concatenate(seq_labels)
    return np.vstack(seq_data), np.concatenate(seq_labels)


def gen_2d_data(target, sample_length, step_size, is_train):
    seq_data, seq_labels = [], []
    for i, t in enumerate(target):
        d, _ = read_data(t, is_train)
        num_matrices = (d.shape[0] - sample_length) // step_size + 1
        for j in range(num_matrices):
            start_index = j * step_size
            end_index = start_index + sample_length
            seq_data.append(d[start_index:end_index])
            seq_labels.append(i)
    seq_data = np.stack(seq_data)

    return seq_data, np.array(seq_labels)


def train(model, optimizer, train_loader):
    model.train()
    acc = AccMectric()
    for data, labels in train_loader:
        x = torch.autograd.Variable(data.cuda())
        y = torch.autograd.Variable(labels.cuda())
        o = model.forward(x)
        loss = torch.nn.CrossEntropyLoss()(o, y)
        acc.update(labels.numpy(), o.data.cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return acc.get()


def validate(model, validate_loader):
    model.eval()
    acc = AccMectric()
    for data, labels in validate_loader:
        x = torch.autograd.Variable(data.cuda())
        o = model(x)
        acc.update(labels.numpy(), o.data.cpu().numpy())
    return acc.get()


if __name__ == "__main__":
    target = [1, 2, 3]
    gen_2d_data(target, 20, 2, True)
