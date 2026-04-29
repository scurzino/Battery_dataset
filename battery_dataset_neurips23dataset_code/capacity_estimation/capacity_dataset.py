import os
import torch
from glob import glob
import numpy as np
from tqdm import tqdm

class CapacityDataset:
    '''
    If you want to use another vendor, just switch paths
    ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict.npz.npy'
    ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict1.npz.npy'
    ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict2.npz.npy'
    ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict3.npz.npy'
    '''
    def __init__(self, all_car_dict_path=os.path.join(os.path.dirname(__file__), '../five_fold_utils/all_car_dict.npz.npy'),
                 ind_ood_car_dict_path=os.path.join(os.path.dirname(__file__), '../five_fold_utils/ind_odd_dict1.npz.npy'),
                 train=True, fold_num=0):
        self.all_car_dict = np.load(all_car_dict_path, allow_pickle=True).item()
        ind_ood_car_dict = np.load(ind_ood_car_dict_path, allow_pickle=True).item()
        self.ind_car_num_list = ind_ood_car_dict['ind_sorted']
        # self.ind_car_num_list = [2, 193, 45, 73, 354]  # used for debug
        self.ood_car_num_list = ind_ood_car_dict['ood_sorted']
        print(len(self.ind_car_num_list))
        print(len(self.ood_car_num_list))
        # self.ood_car_num_list = [186, 204, 349, 236, 136]  # used for debug
        if train:
            car_number = self.ind_car_num_list[:int(fold_num * len(self.ind_car_num_list) / 5)] \
                         + self.ind_car_num_list[int((fold_num + 1) * len(self.ind_car_num_list) / 5):] \
                         + self.ood_car_num_list[:int(fold_num * len(self.ood_car_num_list) / 5)] \
                         + self.ood_car_num_list[int((fold_num + 1) * len(self.ood_car_num_list) / 5):]
        else:  # test
            car_number = self.ind_car_num_list[int(fold_num * len(self.ind_car_num_list) / 5):int(
                             (fold_num + 1) * len(self.ind_car_num_list) / 5)] \
                         + self.ood_car_num_list[int(fold_num * len(self.ood_car_num_list) / 5):int(
                             (fold_num + 1) * len(self.ood_car_num_list) / 5)]

        self.battery_dataset = []

        print('car_number is ', car_number)

        capacity_valid_car_number = []

        print("Loading data")
        for each_num in tqdm(car_number):
            for each_pkl in self.all_car_dict[each_num]:
                train1 = torch.load(each_pkl)
                if train1[1]["capacity"] != 0:
                    self.battery_dataset.append(train1)
                    capacity_valid_car_number.append(train1[1]["car"])
        print("unsorted capacity_valid_car_number",
              len(set(capacity_valid_car_number)), set(capacity_valid_car_number),
              "capacity data point",
              len(capacity_valid_car_number))

    def __len__(self):
        return len(self.battery_dataset)

    def __getitem__(self, idx):
        file = self.battery_dataset[idx]
        return file

class SlidingWindowDataset_forecast(torch.utils.data.Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window : index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window

class SlidingWindowDataset_reconstruct(torch.utils.data.Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window : index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window

