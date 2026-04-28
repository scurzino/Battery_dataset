import json
import os
import pickle
import sys
import time
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from tqdm import tqdm
from model import tasks
from model import dynamic_vae
from utils import to_var, collate, Normalizer, PreprocessNormalizer
from model import dataset

from sklearn.preprocessing import MinMaxScaler, RobustScaler

class Train_fivefold:
    """
    for training
    """

    def __init__(self, args, fold_num=0):
        """
        initialization, load project arguments and create folders
        """
        self.args = args
        time_now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        current_path = os.path.join(self.args.save_model_path, time_now+'_fold%d'%fold_num)
        self.mkdir(current_path)
        self.current_path = current_path
        self.current_epoch = 1
        self.step = 1
        self.loss_dict = OrderedDict()
        self.fold_num = fold_num

        loss_picture_path = os.path.join(current_path, "loss")
        feature_path = os.path.join(current_path, "feature")
        current_model_path = os.path.join(current_path, "model")
        save_feature_path = os.path.join(current_path, "mean")
        result_path = os.path.join(current_path, "result")
        # create folders
        self.mkdir(loss_picture_path)
        self.mkdir(feature_path)
        self.mkdir(current_model_path)
        self.mkdir(result_path)
        self.mkdir(save_feature_path)

        self.args.loss_picture_path = loss_picture_path
        self.args.feature_path = feature_path
        self.args.result_path = result_path
        self.args.save_feature_path = save_feature_path
        self.args.current_path = current_path
        self.args.current_model_path = current_model_path

    @staticmethod
    def mkdir(path):
        """
        create folders
        :param path: path
        """
        if os.path.exists(path):
            print('%s is exist' % path)
        else:
            os.makedirs(path)

    def main(self):
        """
        training
        load training data, preprocessing, create & train & save model, save parameters
        train: normalized data
        model: model
        loss: nll kl label
        rec_error: reconstruct error
        """
        print("Loading data to memory. This may take a few minutes...")
        data_pre = dataset.Dataset("pathtotrain", train=True, fold_num=self.fold_num)
        self.normalizer = Normalizer(dfs=[data_pre[i][0] for i in range(200)],
                                     variable_length=self.args.variable_length)
        train = PreprocessNormalizer(data_pre, normalizer_fn=self.normalizer.norm_func)
        print("Data loaded successfully.")

        self.args.columns = ['volt',
                             'current',
                             'soc',
                             'max_single_volt',
                             'min_single_volt',
                             'max_temp',
                             'min_temp',
                             'timestamp']
        self.data_task = tasks.Task(task_name=self.args.task, columns=self.args.columns)
        params = dict(
            rnn_type=self.args.rnn_type,
            hidden_size=self.args.hidden_size,
            latent_size=self.args.latent_size,
            num_layers=self.args.num_layers,
            bidirectional=self.args.bidirectional,
            kernel_size=self.args.kernel_size,
            nhead=self.args.nhead,
            dim_feedforward=self.args.dim_feedforward,
            variable_length=self.args.variable_length,
            encoder_embedding_size=self.data_task.encoder_dimension,
            decoder_embedding_size=self.data_task.decoder_dimension,
            output_embedding_size=self.data_task.output_dimension)
        # specify model
        if self.args.model_type == "rnn":
            model = to_var(dynamic_vae.DynamicVAE(**params)).float()
        else:
            model = None

        print("model", model)
        # specify optimizer and learning scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs,
                                      eta_min=self.args.cosine_factor * self.args.learning_rate)
        # DataLoader
        data_loader = DataLoader(dataset=train, batch_size=self.args.batch_size, shuffle=True,
                                 num_workers=self.args.jobs, drop_last=False, pin_memory=torch.cuda.is_available(),
                                 collate_fn=collate if self.args.variable_length else None)

        time_start = time.time()
        try:
            p_bar = tqdm(total=len(data_loader) * self.args.epochs, desc='training', ncols=160, mininterval=1,
                         maxinterval=10, miniters=1)
            while self.current_epoch <= self.args.epochs:
                model.train()
                total_loss, total_nll, total_label, total_kl, iteration = 0, 0, 0, 0, 0
                for batch in data_loader:
                    batch_ = to_var(batch[0]).float()
                    seq_lengths = batch[1]['seq_lengths'] if self.args.variable_length else None
                    log_p, mean, log_v, z, mean_pred = model(batch_,
                                                             encoder_filter=self.data_task.encoder_filter,
                                                             decoder_filter=self.data_task.decoder_filter,
                                                             seq_lengths=seq_lengths, noise_scale=self.args.noise_scale)
                    target = self.data_task.target_filter(batch_)

                    nll_loss, kl_loss, kl_weight = self.loss_fn(log_p, target, mean, log_v)
                    self.label_data = tasks.Label(column_name="mileage", training_set=train)
                    label_loss = self.label_data.loss(batch, mean_pred, is_mse=True)
                    loss = (self.args.nll_weight * nll_loss + self.args.latent_label_weight * label_loss + kl_weight *
                            kl_loss / batch_.shape[0])

                    # update parameters
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # calculate loss
                    total_loss += loss.item()
                    total_nll += nll_loss.item()
                    total_label += label_loss.item()
                    total_kl += kl_loss.item() / batch_.shape[0]
                    loss_info = {'mean_loss': total_loss / (1 + iteration), 'nll_loss': total_nll / (1 + iteration),
                                 "label_loss": total_label / (1 + iteration), "kl_loss": total_kl / (1 + iteration)}
                    p_bar.set_postfix(loss_info)
                    p_bar.set_description('training - Epoch %d/%i' % (self.current_epoch, self.args.epochs))

                    # save loss
                    if iteration == len(data_loader) - 1:
                        self.save_loss(loss_info, log_p, target)

                    self.step += 1
                    p_bar.update(1)
                    iteration += 1

                scheduler.step()
                self.current_epoch += 1
            p_bar.close()

        except KeyboardInterrupt:
            print("Caught keyboard interrupt; quit training.")
            pass

        print("Train completed, save information")
        # save model and parameters
        model.eval()
        p_bar = tqdm(total=len(data_loader), desc='saving', ncols=100, mininterval=1, maxinterval=10, miniters=1)
        extract(data_loader, model, self.data_task, self.args.feature_path, p_bar, self.args.noise_scale,
                self.args.variable_length)
        p_bar.close()
        print("The total time consuming: ", time.time() - time_start)
        self.model_result_save(model)
        self.loss_visual()
        print("All parameters have been saved at", self.args.feature_path)

    def model_result_save(self, model):
        """
        save model
        :param model: vae or transformer
        :return:
        """
        model_params = {'train_time_start': self.current_path,
                        'train_time_end': time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                        'args': vars(self.args),
                        'loss': self.loss_dict}
        with open(os.path.join(self.args.current_model_path, 'model_params.json'), 'w') as f:
            json.dump(model_params, f, indent=4)
        model_path = os.path.join(self.args.current_model_path, "model.torch")
        torch.save(model, model_path)
        norm_path = os.path.join(self.args.current_model_path, "norm.pkl")
        with open(norm_path, "wb") as f:
            pickle.dump(self.normalizer, f)

    def loss_fn(self, log_p, target, mean, log_v):
        """
        loss function
        :param log_p: transformed prediction
        :param target: target
        :param mean:
        :param log_v:
        :return: nll_loss, kl_loss, kl_weight
        """
        nll = torch.nn.SmoothL1Loss(reduction='mean')
        nll_loss = nll(log_p, target)
        kl_loss = -0.5 * torch.sum(1 + log_v - mean.pow(2) - log_v.exp())
        kl_weight = self.kl_anneal_function()
        return nll_loss, kl_loss, kl_weight

    def kl_anneal_function(self):
        """
        anneal update function
        """
        if self.args.anneal_function == 'logistic':
            return self.args.anneal0 * float(1 / (1 + np.exp(-self.args.k * (self.step - self.args.x0))))
        elif self.args.anneal_function == 'linear':
            return self.args.anneal0 * min(1, self.step / self.args.x0)
        else:
            return self.args.anneal0

    def loss_visual(self):
        """
        draw loss curves
        """
        if self.args.epochs == 0:
            return
        x = list(self.loss_dict.keys())
        df_loss = pd.DataFrame(dict(self.loss_dict)).T.sort_index()
        mean_loss = df_loss['mean_loss'].values.astype(float)
        nll_loss = df_loss['nll_loss'].values.astype(float)
        label_loss = df_loss['label_loss'].values.astype(float)
        kl_loss = df_loss['kl_loss'].values.astype(float)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(x, mean_loss, 'r.-', label='mean_loss')
        plt.legend()

        plt.subplot(2, 3, 4)
        plt.plot(x, nll_loss, 'bo-', label='nll_loss')
        plt.legend()

        plt.subplot(2, 3, 5)
        plt.plot(x, label_loss, 'bo-', label='label_loss')
        plt.legend()

        plt.subplot(2, 3, 6)
        plt.plot(x, kl_loss, 'bo-', label='kl_loss')
        plt.legend()
        plt.savefig(self.args.loss_picture_path + '/' + 'loss.png')
        plt.close('all')

    def save_loss(self, loss_info, log_p, target):
        """
        save loss
        """
        self.loss_dict[str(self.current_epoch)] = loss_info
        n_image = log_p.shape[-1]
        for i in range(n_image):
            plt.subplot(n_image, 1, i + 1)
            plt.plot(log_p[0, :, i].cpu().detach().numpy(), 'y',
                     label='lp-' + str(self.current_epoch))
            plt.plot(target[0, :, i].cpu().detach().numpy(), 'c',
                     label='tg-' + str(self.current_epoch))
            plt.legend()
        loss_path = os.path.join(self.args.loss_picture_path, "%i_epoch.jpg" % self.current_epoch)
        plt.savefig(loss_path)
        plt.close('all')

    def getmodelparams(self):
        return os.path.join(self.args.current_model_path, 'model_params.json')


def save_features_info(feature_path, batch, iteration, log_p, mean, target):
    """
    save features
    """
    mse = torch.nn.MSELoss(reduction='mean')
    dict_path = os.path.join(feature_path, "%i_label.file" % iteration)
    with open(dict_path, "wb") as f:
        rec_error = [float(mse(log_p[i], target[i])) for i in range(batch[0].shape[0])]
        batch[1].update({'rec_error': rec_error})
        torch.save(batch[1], f)
    mean_path = os.path.join(feature_path, "%i_npy.npy" % iteration)
    np_mean = mean.data.cpu().numpy()
    np.save(mean_path, np_mean)


class Train_spacecraft:
    def __init__(self, args):

        self.args = args
        time_now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        # time_now = "trial"
        current_path = os.path.join(self.args.save_model_path, time_now)
        self.mkdir(current_path)
        self.current_path = current_path
        self.current_epoch = 1
        self.step = 1
        self.loss_dict = OrderedDict()

        loss_picture_path = os.path.join(current_path, "loss")
        feature_path = os.path.join(current_path, "feature")
        current_model_path = os.path.join(current_path, "model")
        save_feature_path = os.path.join(current_path, "mean")
        result_path = os.path.join(current_path, "result")

        self.mkdir(loss_picture_path)
        self.mkdir(feature_path)
        self.mkdir(current_model_path)
        self.mkdir(result_path)
        self.mkdir(save_feature_path)

        self.args.loss_picture_path = loss_picture_path
        self.args.feature_path = feature_path
        self.args.result_path = result_path
        self.args.save_feature_path = save_feature_path
        self.args.current_path = current_path
        self.args.current_model_path = current_model_path

    @staticmethod
    def mkdir(path):
        if os.path.exists(path):
            print('%s is exist' % path)
        else:
            os.makedirs(path)

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
        print("Loading data to memory. This may take a few minutes...")
        (x_train, _), (x_test, y_test) = self.get_data(self.args.project, normalize=True)
        if reconstruct:
            train_dataset = dataset.SlidingWindowDataset_reconstruct(x_train, self.args.window_size, target_dim=[0])
            test_dataset = dataset.SlidingWindowDataset_reconstruct(x_test, self.args.window_size, target_dim=[0])
        else: # forecast
            train_dataset = dataset.SlidingWindowDataset_forecast(x_train, self.args.window_size, target_dim=[0])
            test_dataset = dataset.SlidingWindowDataset_forecast(x_test, self.args.window_size, target_dim=[0])

        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_dataset, self.args.batch_size, self.args.val_split, shuffle=True, test_dataset=test_dataset
        )
        print("Data loaded successfully.")

        self.data_task = tasks.Task(task_name=self.args.task, columns='auto')
        params = dict(
            rnn_type=self.args.rnn_type,
            hidden_size=self.args.hidden_size,
            latent_size=self.args.latent_size,
            num_layers=self.args.num_layers,
            bidirectional=self.args.bidirectional,
            kernel_size=self.args.kernel_size,
            nhead=self.args.nhead,
            dim_feedforward=self.args.dim_feedforward,
            variable_length=self.args.variable_length,
            encoder_embedding_size=self.data_task.encoder_dimension,
            decoder_embedding_size=self.data_task.decoder_dimension,
            output_embedding_size=self.data_task.output_dimension)

        if self.args.model_type == "rnn":
            model = to_var(dynamic_vae.DynamicVAE_plus(**params)).float()
        else:
            model = None

        print("model", model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.learning_rate, weight_decay=1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.epochs,
                                      eta_min=self.args.cosine_factor * self.args.learning_rate)

        time_start = time.time()
        best_val_total_loss = 1000000
        try:
            p_bar = tqdm(total=len(train_loader) * self.args.epochs, desc='training', ncols=160, mininterval=1,
                         maxinterval=10, miniters=1)
            while self.current_epoch <= self.args.epochs:
                model.train()
                total_loss, total_nll, total_label, total_kl, iteration = 0, 0, 0, 0, 0
                for batch in train_loader:
                    batch_ = to_var(batch[0]).float()
                    seq_lengths = None
                    log_p, mean, log_v, z, mean_pred = model(batch_,
                                                             encoder_filter=self.data_task.encoder_filter,
                                                             decoder_filter=self.data_task.decoder_filter,
                                                             seq_lengths=seq_lengths, noise_scale=self.args.noise_scale)
                    if reconstruct:
                        target = self.data_task.target_filter(batch_)
                        nll_loss, kl_loss, kl_weight = self.loss_fn(log_p, target, mean, log_v)
                        loss = (self.args.nll_weight * nll_loss + kl_weight * kl_loss / batch_.shape[0])
                    else:
                        target = to_var(batch[1][:, :, 0:self.data_task.output_dimension]).float()
                        nll_loss, kl_loss, kl_weight = self.loss_fn(log_p[:, -1:, :], target, mean, log_v)
                        loss = (self.args.nll_weight * nll_loss + kl_weight * kl_loss / batch_.shape[0])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    total_nll += nll_loss.item()
                    total_kl += kl_loss.item() / batch_.shape[0]
                    loss_info = {'mean_loss': total_loss / (1 + iteration), 'nll_loss': total_nll / (1 + iteration),
                                 "kl_loss": total_kl / (1 + iteration)}
                    p_bar.set_postfix(loss_info)
                    p_bar.set_description('training - Epoch %d/%i' % (self.current_epoch, self.args.epochs))

                    if iteration == len(train_loader) - 1:
                        self.save_loss(loss_info, log_p, target)

                    self.step += 1
                    p_bar.update(1)
                    iteration += 1

                # validation
                model.eval()
                val_total_loss, val_total_nll, val_total_label, val_total_kl, val_iteration = 0, 0, 0, 0, 0
                for batch in val_loader:
                    batch_ = to_var(batch[0]).float()
                    seq_lengths = batch[1]['seq_lengths'] if self.args.variable_length else None
                    log_p, mean, log_v, z, mean_pred = model(batch_,
                                                             encoder_filter=self.data_task.encoder_filter,
                                                             decoder_filter=self.data_task.decoder_filter,
                                                             seq_lengths=seq_lengths, noise_scale=self.args.noise_scale)
                    if reconstruct:
                        target = self.data_task.target_filter(batch_)
                        val_nll_loss, val_kl_loss, val_kl_weight = self.loss_fn(log_p, target, mean, log_v)
                    else:
                        target = to_var(batch[1][:, :, 0:self.data_task.output_dimension]).float()
                        val_nll_loss, val_kl_loss, val_kl_weight = self.loss_fn(log_p[:, -1:, :], target, mean, log_v)

                    val_loss = (self.args.nll_weight * val_nll_loss + val_kl_weight * val_kl_loss / batch_.shape[0])

                    val_total_loss += val_loss.item()
                    val_total_nll += val_nll_loss.item()
                    val_total_kl += val_kl_loss.item() / batch_.shape[0]
                    val_iteration += 1

                # test
                model.eval()
                test_total_loss, test_total_nll, test_total_label, test_total_kl, test_iteration = 0, 0, 0, 0, 0
                for batch in test_loader:
                    batch_ = to_var(batch[0]).float()
                    seq_lengths = batch[1]['seq_lengths'] if self.args.variable_length else None
                    log_p, mean, log_v, z, mean_pred = model(batch_,
                                                             encoder_filter=self.data_task.encoder_filter,
                                                             decoder_filter=self.data_task.decoder_filter,
                                                             seq_lengths=seq_lengths, noise_scale=self.args.noise_scale)
                    if reconstruct:
                        target = self.data_task.target_filter(batch_)
                        test_nll_loss, test_kl_loss, test_kl_weight = self.loss_fn(log_p, target, mean, log_v)
                    else:
                        target = to_var(batch[1][:, :, 0:self.data_task.output_dimension]).float()
                        test_nll_loss, test_kl_loss, test_kl_weight = self.loss_fn(log_p[:, -1:, :], target, mean, log_v)


                    test_loss = (self.args.nll_weight * test_nll_loss + test_kl_weight * test_kl_loss / batch_.shape[0])

                    test_total_loss += test_loss.item()
                    test_total_nll += test_nll_loss.item()
                    test_total_kl += test_kl_loss.item() / batch_.shape[0]

                    test_iteration += 1

                # save best
                if val_total_loss < best_val_total_loss:
                    print("saving best model")
                    print("val_total_loss",
                          val_total_loss / (1 + val_iteration),
                          val_total_nll / (1 + val_iteration),
                          val_total_kl / (1 + val_iteration))
                    print("test_total_loss",
                          test_total_loss / (1 + test_iteration),
                          test_total_nll / (1 + test_iteration),
                          test_total_kl / (1 + test_iteration))
                    model_path = os.path.join(self.args.current_model_path, "model_best.torch")
                    torch.save(model, model_path)
                    best_val_total_loss = val_total_loss
                scheduler.step()
                self.current_epoch += 1
            p_bar.close()

        except KeyboardInterrupt:
            print("Caught keyboard interrupt; quit training.")
            pass

        print("Train completed, save information")
        # load best model
        best_model_path = os.path.join(self.args.current_model_path, "model_best.torch")
        # model = model.load_state_dict(torch.load(best_model_path))
        model = to_var(torch.load(best_model_path)).float()

        model.eval()
        full_train_loader, val_loader, test_loader = self.create_data_loaders(
            train_dataset, self.args.batch_size, 0, shuffle=True, test_dataset=test_dataset
        )
        p_bar = tqdm(total=len(full_train_loader), desc='saving', ncols=100, mininterval=1, maxinterval=10, miniters=1)
        new_extract(full_train_loader, model, self.data_task, self.args.feature_path, p_bar, self.args.noise_scale,
                self.args.variable_length, save_name="train", reconstruct=reconstruct)
        p_bar.close()
        print("The total time consuming：", time.time() - time_start)
        self.model_result_save(model)
        self.loss_visual()
        print("All parameters have been saved at", self.args.feature_path)

    def model_result_save(self, model):
        """
        :param model: vae or transformer
        :return:
        """
        model_params = {'train_time_start': self.current_path,
                        'train_time_end': time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                        'args': vars(self.args),
                        'loss': self.loss_dict}
        with open(os.path.join(self.args.current_model_path, 'model_params.json'), 'w') as f:
            json.dump(model_params, f, indent=4)
        model_path = os.path.join(self.args.current_model_path, "model.torch")
        torch.save(model, model_path)

    def loss_fn(self, log_p, target, mean, log_v):
        nll = torch.nn.SmoothL1Loss(reduction='mean')
        nll_loss = nll(log_p, target)
        kl_loss = -0.5 * torch.sum(1 + log_v - mean.pow(2) - log_v.exp())
        kl_weight = self.kl_anneal_function()
        return nll_loss, kl_loss, kl_weight

    def kl_anneal_function(self):
        if self.args.anneal_function == 'logistic':
            return self.args.anneal0 * float(1 / (1 + np.exp(-self.args.k * (self.step - self.args.x0))))
        elif self.args.anneal_function == 'linear':
            return self.args.anneal0 * min(1, self.step / self.args.x0)
        else:
            return self.args.anneal0

    def loss_visual(self):
        if self.args.epochs == 0:
            return
        x = list(self.loss_dict.keys())
        df_loss = pd.DataFrame(dict(self.loss_dict)).T.sort_index()
        mean_loss = df_loss['mean_loss'].values.astype(float)
        nll_loss = df_loss['nll_loss'].values.astype(float)
        kl_loss = df_loss['kl_loss'].values.astype(float)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(x, mean_loss, 'r.-', label='mean_loss')
        plt.legend()

        plt.subplot(2, 3, 4)
        plt.plot(x, nll_loss, 'bo-', label='nll_loss')
        plt.legend()

        plt.subplot(2, 3, 6)
        plt.plot(x, kl_loss, 'bo-', label='kl_loss')
        plt.legend()
        plt.savefig(self.args.loss_picture_path + '/' + 'loss.png')
        plt.close('all')

    def save_loss(self, loss_info, log_p, target):
        self.loss_dict[str(self.current_epoch)] = loss_info
        n_image = log_p.shape[-1]
        for i in range(n_image):
            plt.subplot(n_image, 1, i + 1)
            plt.plot(log_p[0, :, i].cpu().detach().numpy(), 'y',
                     label='lp-' + str(self.current_epoch))
            plt.plot(target[0, :, i].cpu().detach().numpy(), 'c',
                     label='tg-' + str(self.current_epoch))
            plt.legend()
        loss_path = os.path.join(self.args.loss_picture_path, "%i_epoch.jpg" % self.current_epoch)
        plt.savefig(loss_path)
        plt.close('all')

    def getmodelparams(self):
        return os.path.join(self.args.current_model_path, 'model_params.json')



def extract(data_loader, model, data_task, feature_path, p_bar, noise_scale, variable_length):
    """
    extract features
    """
    iteration = 0
    for batch in data_loader:
        batch_ = to_var(batch[0]).float()
        seq_lengths = batch[1]['seq_lengths'] if variable_length else None
        log_p, mean, log_v, z, mean_pred = model(batch_, encoder_filter=data_task.encoder_filter,
                                                 decoder_filter=data_task.decoder_filter,
                                                 seq_lengths=seq_lengths, noise_scale=noise_scale)
        target = data_task.target_filter(batch_)
        save_features_info(feature_path, batch, iteration, log_p, mean, target)
        p_bar.update(1)
        iteration += 1

def new_extract(data_loader, model, data_task, feature_path, p_bar, noise_scale, variable_length, save_name=None, reconstruct=None):
    assert reconstruct is not None
    mse = torch.nn.MSELoss(reduction='mean')

    rec_error_list = []
    for batch in data_loader:
        batch_ = to_var(batch[0]).float()

        seq_lengths = batch[1]['seq_lengths'] if variable_length else None
        log_p, mean, log_v, z, mean_pred = model(batch_, encoder_filter=data_task.encoder_filter,
                                                 decoder_filter=data_task.decoder_filter,
                                                 seq_lengths=seq_lengths, noise_scale=noise_scale)
        if reconstruct:
            target = data_task.target_filter(batch_)
            rec_error = (log_p - target) ** 2
            rec_error_list.append(rec_error.detach().cpu().numpy().reshape(-1))
        else:  # forecast
            target = to_var(batch[1][:, :, 0:data_task.output_dimension]).float()

            rec_error = (log_p[:, -1:, :] - target) ** 2
            rec_error_list.append(rec_error.detach().cpu().numpy().reshape(-1))
    rec_error_list = np.hstack(rec_error_list)
    np.save(os.path.join(feature_path, save_name+".npy"), rec_error_list)


if __name__ == '__main__':
    import argparse

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(description='Train Example')
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(os.path.dirname(os.getcwd()), './params.json'))

    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(file))
        args = parser.parse_args(namespace=p_args)
    print("Loaded configs at %s" % args.config_path)
    print("args", args)
    Train(args).main()
