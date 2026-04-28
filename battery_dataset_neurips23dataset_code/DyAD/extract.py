import json
import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import collate
from model import dataset
from train import extract, new_extract
from utils import to_var, collate, Normalizer, PreprocessNormalizer
from model import tasks
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

class Extraction:
    """
    feature extraction
    """

    def __init__(self, args, fold_num=0):
        """
        :param project: class model.projects.Project object
        """
        self.args = args
        self.fold_num = fold_num

    def main(self):
        """
        test: normalized test data
        task: task, e.g. EvTask
        model: model
        """
        model_params_path = os.path.join(self.args.current_model_path, "model_params.json")
        with open(model_params_path, 'r') as load_f:
            prams_dict = json.load(load_f)
        model_params = prams_dict['args']
        start_time = time.time()
        data_pre = dataset.Dataset("pathtotest", train=False, fold_num=self.fold_num)
        self.normalizer = pickle.load(open(os.path.join(self.args.current_model_path, "norm.pkl"), 'rb'))
        test = PreprocessNormalizer(data_pre, normalizer_fn=self.normalizer.norm_func)

        task = tasks.Task(task_name=model_params["task"], columns=model_params["columns"])

        # load checkpoint
        model_torch = os.path.join(model_params["current_model_path"], "model.torch")
        model = to_var(torch.load(model_torch)).float()
        model.encoder_filter = task.encoder_filter
        model.decoder_filter = task.decoder_filter
        model.noise_scale = model_params["noise_scale"]
        data_loader = DataLoader(dataset=test, batch_size=model_params["batch_size"], shuffle=True,
                                 num_workers=model_params["jobs"], drop_last=False,
                                 pin_memory=torch.cuda.is_available(),
                                 collate_fn=collate if model_params["variable_length"] else None)

        print("sliding windows dataset length is: ", len(test))
        print("model", model)

        # extact feature
        model.eval()
        p_bar = tqdm(total=len(data_loader), desc='saving', ncols=100, mininterval=1, maxinterval=10, miniters=1)
        extract(data_loader, model, task, model_params["save_feature_path"], p_bar, model_params["noise_scale"],
                model_params["variable_length"])
        p_bar.close()
        print("Feature extraction of all test saved at", model_params["save_feature_path"])
        print("The total time consuming：", time.time() - start_time)


class Extraction_spacecraft:
    """
    feature extract
    """

    def __init__(self, args):
        """
        :param project: class model.projects.Project object
        """
        self.args = args

    def normalize_data(self, data, scaler=None):
        data = np.asarray(data, dtype=np.float32)
        if np.any(sum(np.isnan(data))):
            data = np.nan_to_num(data)

        if scaler is None:
            scaler = MinMaxScaler()
            scaler.fit(data)
        data = scaler.transform(data)
        print("Data normalized")

        return data, scaler

    def get_data(self, dataset, max_train_size=None, max_test_size=None,
                 normalize=False, spec_res=False, train_start=0, test_start=0):
        """
        Get data from pkl files
        return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
        Method from OmniAnomaly (https://github.com/NetManAIOps/OmniAnomaly)
        """
        prefix = "msl_smap_dataset"
        if dataset in ["MSL", "SMAP"]:
            prefix += "/processed"
        if max_train_size is None:
            train_end = None
        else:
            train_end = train_start + max_train_size
        if max_test_size is None:
            test_end = None
        else:
            test_end = test_start + max_test_size
        print("load data of:", dataset)
        print("train: ", train_start, train_end)
        print("test: ", test_start, test_end)
        if dataset == "SMAP":
            x_dim = 25
        elif dataset == "MSL":
            x_dim = 55
        f = open(os.path.join(prefix, dataset + "_train.pkl"), "rb")
        train_data = pickle.load(f).reshape((-1, x_dim))[train_start:train_end, :]
        f.close()
        try:
            f = open(os.path.join(prefix, dataset + "_test.pkl"), "rb")
            test_data = pickle.load(f).reshape((-1, x_dim))[test_start:test_end, :]
            f.close()
        except (KeyError, FileNotFoundError):
            test_data = None
        try:
            f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
            test_label = pickle.load(f).reshape((-1))[test_start:test_end]
            f.close()
        except (KeyError, FileNotFoundError):
            test_label = None

        if normalize:
            train_data, scaler = self.normalize_data(train_data, scaler=None)
            test_data, _ = self.normalize_data(test_data, scaler=scaler)

        print("train set shape: ", train_data.shape)
        print("test set shape: ", test_data.shape)
        print("test set label shape: ", None if test_label is None else test_label.shape)
        return (train_data, None), (test_data, test_label)

    def create_data_loaders(self, train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
        train_loader, val_loader, test_loader = None, None, None
        if val_split == 0.0:
            print(f"train_size: {len(train_dataset)}")
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        else:
            dataset_size = len(train_dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(val_split * dataset_size))
            if shuffle:
                np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
            valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

            print(f"train_size: {len(train_indices)}")
            print(f"validation_size: {len(val_indices)}")

        if test_dataset is not None:
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            print(f"test_size: {len(test_dataset)}")

        return train_loader, val_loader, test_loader


    def main(self, reconstruct=True):
        model_params_path = os.path.join(self.args.current_model_path, "model_params.json")
        with open(model_params_path, 'r') as load_f:
            prams_dict = json.load(load_f)
        model_params = prams_dict['args']
        start_time = time.time()

        (x_train, _), (x_test, y_test) = self.get_data(self.args.project, normalize=True)
        if reconstruct:
            train_dataset = dataset.SlidingWindowDataset_reconstruct(x_train, self.args.window_size, target_dim=[0])
            test_dataset = dataset.SlidingWindowDataset_reconstruct(x_test, self.args.window_size, target_dim=[0])
        else:
            train_dataset = dataset.SlidingWindowDataset_forecast(x_train, self.args.window_size, target_dim=[0])
            test_dataset = dataset.SlidingWindowDataset_forecast(x_test, self.args.window_size, target_dim=[0])
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_dataset, self.args.batch_size, self.args.val_split, shuffle=True, test_dataset=test_dataset
        )
        print("Data loaded successfully.")

        task = tasks.Task(task_name=model_params["task"], columns='auto')

        # open model file
        model_torch = os.path.join(model_params["current_model_path"], "model_best.torch")
        model = to_var(torch.load(model_torch)).float()
        model.encoder_filter = task.encoder_filter
        model.decoder_filter = task.decoder_filter
        model.noise_scale = model_params["noise_scale"]

        print("sliding windows dataset length is: ", len(test_dataset))
        print("model", model)

        # extract feature
        model.eval()
        p_bar = tqdm(total=len(test_loader), desc='saving', ncols=100, mininterval=1, maxinterval=10, miniters=1)
        new_extract(test_loader, model, task, model_params["save_feature_path"], p_bar, model_params["noise_scale"],
                model_params["variable_length"], save_name="test", reconstruct=reconstruct)
        p_bar.close()
        print("Feature extraction of all test saved at", model_params["save_feature_path"])
        print("The total time consuming：", time.time() - start_time)

if __name__ == '__main__':
    import argparse

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(description='Train Example')
    parser.add_argument('--current_model_path', type=str,
                        default='/home/user/cleantest/2021-12-04-15-19-38/model/')
    args = parser.parse_args()
    Extraction(args).main()
