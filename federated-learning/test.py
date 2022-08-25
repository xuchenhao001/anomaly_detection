import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# https://github.com/Bingmang/kddcup99-cnn/blob/master/kdd-pytorch.ipynb
class KddData(object):

    def __init__(self, batch_size):
        kddcup99 = datasets.fetch_kddcup99(data_home="../data/kdd99/")
        self._encoder = {
            'protocal': LabelEncoder(),
            'service': LabelEncoder(),
            'flag': LabelEncoder(),
            'label': LabelEncoder()
        }
        self.batch_size = batch_size
        data_X, data_y = self.__encode_data(kddcup99.data, kddcup99.target)
        self.train_dataset, self.test_dataset = self.__split_data_to_tensor(data_X, data_y)
        self.train_dataloader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, self.batch_size, shuffle=True)

    """将数据中字符串部分转换为数字，并将输入的41维特征转换为8*8的矩阵"""

    def __encode_data(self, data_X, data_y):
        self._encoder['protocal'].fit(list(set(data_X[:, 1])))
        self._encoder['service'].fit(list(set(data_X[:, 2])))
        self._encoder['flag'].fit((list(set(data_X[:, 3]))))
        self._encoder['label'].fit(list(set(data_y)))
        data_X[:, 1] = self._encoder['protocal'].transform(data_X[:, 1])
        data_X[:, 2] = self._encoder['service'].transform(data_X[:, 2])
        data_X[:, 3] = self._encoder['flag'].transform(data_X[:, 3])
        data_X = np.pad(data_X, ((0, 0), (0, 64 - len(data_X[0]))), 'constant').reshape(-1, 1, 8, 8)
        data_y = self._encoder['label'].transform(data_y)
        return data_X, data_y

    """将数据拆分为训练集和测试集，并转换为TensorDataset对象"""

    def __split_data_to_tensor(self, data_X, data_y):
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3)
        train_dataset = TensorDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.int))
        )
        test_dataset = TensorDataset(
            torch.from_numpy(X_test.astype(np.float32)),
            torch.from_numpy(y_test.astype(np.int))
        )
        return train_dataset, test_dataset

    """接受一个数组进行解码"""

    def decode(self, data, label=False):
        if not label:
            _data = list(data)
            _data[1] = self._encoder['protocal'].inverse_transform([_data[1]])[0]
            _data[2] = self._encoder['service'].inverse_transform([_data[2]])[0]
            _data[2] = self._encoder['flag'].inverse_transform([_data[3]])[0]
            return _data
        return self._encoder['label'].inverse_transform(data)

    def encode(self, data, label=False):
        if not label:
            _data = list(data)
            _data[1] = self._encoder['protocal'].transform([_data[1]])[0]
            _data[2] = self._encoder['service'].transform([_data[2]])[0]
            _data[3] = self._encoder['flag'].transform([_data[3]])[0]
            return _data
        return self._encoder['label'].transform([data])[0]

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_dim, n_class):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 6, 3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.Linear(144, 512),
            nn.Linear(512, 256),
            nn.Linear(256, n_class)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# 神经网络参数
batch_size = 128
learning_rate = 1e-2
num_epoches = 20
USE_GPU = torch.cuda.is_available()

dataset = KddData(batch_size)
model = CNN(1, 23)


def train():
    global model

    if USE_GPU:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(num_epoches):
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(dataset.train_dataloader, 1):
            img, label = data
            if USE_GPU:
                img = img.cuda()
                label = label.cuda()
            img = Variable(img)
            label = Variable(label)
            # 向前传播
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            accuracy = (pred == label).float().mean()
            running_acc += num_correct.item()
            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(dataset.train_dataset)), running_acc / (len(
                dataset.train_dataset))))
        model.eval()
        eval_loss = 0
        eval_acc = 0
        for data in dataset.test_dataloader:
            img, label = data
            if USE_GPU:
                img = Variable(img, volatile=True).cuda()
                label = Variable(label, volatile=True).cuda()
            else:
                img = Variable(img, volatile=True)
                label = Variable(label, volatile=True)
            out = model(img)
            loss = criterion(out, label)
            eval_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            eval_acc += num_correct.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            dataset.test_dataset)), eval_acc / (len(dataset.test_dataset))))
        print()


train()

