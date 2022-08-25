import logging
import os
import torch

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.sampling import get_user_indices
from utils.util import ColoredLogger

logging.setLoggerClass(ColoredLogger)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logger = logging.getLogger(__file__)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.targets = torch.Tensor([self.dataset.targets[idx] for idx in idxs])

    def classes(self):
        return torch.unique(self.targets)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        data, label = self.dataset[self.idxs[item]]
        return data, label


class LocalDataset:
    def __init__(self):
        self.initialized = False
        self.dataset_name = ""
        self.dataset_train = None
        self.dataset_test = None
        self.image_shape = None
        self.dict_users = None
        self.test_users = None
        self.dis_acc_test = False

    def init_local_dataset(self, dataset_name, dataset_size, num_users, dis_acc_test):
        dataset_train = None
        dataset_test = None
        real_path = os.path.dirname(os.path.realpath(__file__))
        # load dataset and split users
        if dataset_name == 'kdd99':
            data_path = os.path.join(real_path, "../../data/kdd99/")
            kddcup99 = datasets.fetch_kddcup99(data_home=data_path)
            self._encoder = {
                'protocal': LabelEncoder(),
                'service': LabelEncoder(),
                'flag': LabelEncoder(),
                'label': LabelEncoder()
            }
            data_X, data_y = self.encode_data(kddcup99.data, kddcup99.target)
            dataset_train, dataset_test = self.split_datasets(data_X, data_y)

        dict_users, test_users = get_user_indices(dataset_name, dataset_train, dataset_test, dataset_size, num_users)

        self.dataset_name = dataset_name
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.image_shape = dataset_train[0][0].shape
        self.dict_users = dict_users
        self.test_users = test_users
        self.dis_acc_test = dis_acc_test
        self.initialized = True

    def load_train_dataset(self, idx, local_bs):
        split_ds = DatasetSplit(self.dataset_train, self.dict_users[idx])
        return DataLoader(split_ds, batch_size=local_bs, shuffle=True)

    def load_test_dataset(self, idxs, local_test_bs):
        if self.dis_acc_test:
            split_ds = DatasetSplit(self.dataset_test, idxs)
            return DataLoader(split_ds, batch_size=local_test_bs)
        else:
            return DataLoader(self.dataset_test, batch_size=local_test_bs)

    """将数据拆分为训练集和测试集，并转换为TensorDataset对象"""
    def split_datasets(self, data_X, data_y):
        X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.3)
        train_dataset = TorchDataset(
            torch.from_numpy(X_train.astype(np.float32)),
            torch.from_numpy(y_train.astype(np.int))
        )
        test_dataset = TorchDataset(
            torch.from_numpy(X_test.astype(np.float32)),
            torch.from_numpy(y_test.astype(np.int))
        )
        return train_dataset, test_dataset

    """将数据中字符串部分转换为数字，并将输入的41维特征转换为8*8的矩阵"""
    def encode_data(self, data_X, data_y):
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


class TorchDataset(TensorDataset):
    def __init__(self, *tensors: Tensor):
        super().__init__(*tensors)
        self.targets = [item[1] for item in self]
