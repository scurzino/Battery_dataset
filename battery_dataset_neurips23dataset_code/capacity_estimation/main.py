import argparse
import json
import os
import sys
import torch
from capacity_dataset import CapacityDataset
from utils import to_var, collate, Normalizer, PreprocessNormalizer
from model import LSTMNet, MLP, GatedCNN
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from utils import build_loc_net, get_fc_graph_struc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch capacity estimation')
    parser.add_argument('--fold_num', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model', type=str, default='LSTMNet')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--load_saved_dataset', action='store_true')


    args = parser.parse_args()

    print("args", args)

    # load dataset
    if args.load_saved_dataset:
        with open(f'saved_dataset/train_dataset_fold_{args.fold_num}.pkl', 'rb') as f:
            train_dataset = pickle.load(f)
        with open(f'saved_dataset/test_dataset_{args.fold_num}.pkl', 'rb') as f:
            test_dataset = pickle.load(f)
    else:
        train_pre = CapacityDataset(train=True, fold_num=args.fold_num)
        test_pre = CapacityDataset(train=False, fold_num=args.fold_num)

        normalizer = Normalizer(dfs=[train_pre[i][0] for i in range(200)],
                                     variable_length=False)
        train_dataset = PreprocessNormalizer(train_pre, normalizer_fn=normalizer.norm_func)
        test_dataset = PreprocessNormalizer(test_pre, normalizer_fn=normalizer.norm_func)
        os.makedirs("saved_dataset", exist_ok=True)
        with open(f'saved_dataset/train_dataset_fold_{args.fold_num}.pkl', 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(f'saved_dataset/test_dataset_{args.fold_num}.pkl', 'wb') as f:
            pickle.dump(test_dataset, f)

    # LSTM
    if args.model == "LSTMNet":
        model = LSTMNet(input_dim=8, hidden_dim=32, output_dim=1).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, batch_data in enumerate(tqdm(train_loader)):
                data = to_var(batch_data[0].float())
                capacity = to_var(batch_data[1]['capacity'].float())
                output = model(data)
                loss = criterion(output.reshape(-1), capacity.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss+=loss.item() * data.shape[0]

            print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, args.num_epochs, epoch_loss/len(train_dataset)))

            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                    data = to_var(batch_data[0].float())
                    capacity = to_var(batch_data[1]['capacity'].float())
                    output = model(data)
                    loss = criterion(output.reshape(-1), capacity.reshape(-1))
                    test_loss+=loss.item() * data.shape[0]
            print(
                'Epoch [{}/{}], Test Loss: {:.4f}'.format(epoch + 1, args.num_epochs, test_loss/len(test_dataset)))

    elif args.model == "XGBoost":
        # XGBoost
        train_data_list = [train_dataset.__getitem__(i)[0] for i in range(len(train_dataset))]
        train_labels_list = [train_dataset.__getitem__(i)[1]['capacity'] for i in range(len(train_dataset))]
        train_data = np.array(train_data_list).reshape(len(train_data_list), -1)
        train_labels = np.array(train_labels_list)

        test_data_list = [test_dataset.__getitem__(i)[0] for i in range(len(test_dataset))]
        test_labels_list = [test_dataset.__getitem__(i)[1]['capacity'] for i in range(len(test_dataset))]
        test_data = np.array(test_data_list).reshape(len(test_data_list), -1)
        test_labels = np.array(test_labels_list)

        print(train_data.shape, test_data.shape)

        dtrain = xgb.DMatrix(train_data, label=train_labels)
        dtest = xgb.DMatrix(test_data, label=test_labels)

        params = {
            'objective': 'reg:squarederror',
            'eta': 0.1,
            'max_depth': 3,
            'eval_metric': 'rmse'
        }

        num_rounds = args.num_epochs
        model = xgb.train(params, dtrain, num_rounds)

        train_predictions = model.predict(dtrain)
        test_predictions = model.predict(dtest)

        print("Train RMSE:", np.sqrt(np.mean((train_predictions - train_labels) ** 2)))
        print("Test RMSE:", np.sqrt(np.mean((test_predictions - test_labels) ** 2)))

    elif args.model == "MEAN":
        train_data_list = [train_dataset.__getitem__(i)[0] for i in range(len(train_dataset))]
        train_labels_list = [train_dataset.__getitem__(i)[1]['capacity'] for i in range(len(train_dataset))]
        train_data = np.array(train_data_list).reshape(len(train_data_list), -1)
        train_labels = np.array(train_labels_list)

        test_data_list = [test_dataset.__getitem__(i)[0] for i in range(len(test_dataset))]
        test_labels_list = [test_dataset.__getitem__(i)[1]['capacity'] for i in range(len(test_dataset))]
        test_data = np.array(test_data_list).reshape(len(test_data_list), -1)
        test_labels = np.array(test_labels_list)

        print(train_data.shape, test_data.shape)
        test_predictions = np.ones_like(test_labels)*np.mean(train_labels)

        print("Test RMSE:", np.sqrt(np.mean((test_predictions - test_labels) ** 2)))

    elif args.model == "RandomForest":
        train_data_list = [train_dataset.__getitem__(i)[0] for i in range(len(train_dataset))]
        train_labels_list = [train_dataset.__getitem__(i)[1]['capacity'] for i in range(len(train_dataset))]
        train_data = np.array(train_data_list).reshape(len(train_data_list), -1)
        train_labels = np.array(train_labels_list)

        test_data_list = [test_dataset.__getitem__(i)[0] for i in range(len(test_dataset))]
        test_labels_list = [test_dataset.__getitem__(i)[1]['capacity'] for i in range(len(test_dataset))]
        test_data = np.array(test_data_list).reshape(len(test_data_list), -1)
        test_labels = np.array(test_labels_list)

        model = RandomForestRegressor(n_estimators=10, random_state=0, n_jobs=10, max_depth=4)

        model.fit(train_data, train_labels)

        train_predictions = model.predict(train_data)
        train_rmse = np.sqrt(mean_squared_error(train_labels, train_predictions))
        print("Train RMSE:", train_rmse)

        test_predictions = model.predict(test_data)
        test_rmse = np.sqrt(mean_squared_error(test_labels, test_predictions))
        print("Test RMSE:", test_rmse)

    elif args.model == "MLP":
        model = MLP(input_dim=128*8, hidden_dim=32, output_dim=1).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, batch_data in enumerate(tqdm(train_loader)):
                data = to_var(batch_data[0].float())
                capacity = to_var(batch_data[1]['capacity'].float())
                output = model(data)
                loss = criterion(output.reshape(-1), capacity.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * data.shape[0]

            print(
                'Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, args.num_epochs, epoch_loss / len(train_dataset)))

            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                    data = to_var(batch_data[0].float())
                    capacity = to_var(batch_data[1]['capacity'].float())
                    output = model(data)
                    loss = criterion(output.reshape(-1), capacity.reshape(-1))
                    test_loss += loss.item() * data.shape[0]
            print(
                'Epoch [{}/{}], Test Loss: {:.4f}'.format(epoch + 1, args.num_epochs, test_loss / len(test_dataset)))

    elif args.model == "GatedCNN":
        model = GatedCNN(seq_len=128,
                         n_layers=4,
                         kernel=[5, 5],
                         out_chs=4,
                         res_block_count=2,
                         ans_size=1).cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        for epoch in range(args.num_epochs):
            model.train()
            epoch_loss = 0
            for batch_idx, batch_data in enumerate(tqdm(train_loader)):
                data = to_var(batch_data[0].float())
                capacity = to_var(batch_data[1]['capacity'].float())
                output = model(data)
                loss = criterion(output.reshape(-1), capacity.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * data.shape[0]

            print(
                'Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch + 1, args.num_epochs, epoch_loss / len(train_dataset)))

            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tqdm(test_loader)):
                    data = to_var(batch_data[0].float())
                    capacity = to_var(batch_data[1]['capacity'].float())
                    output = model(data)
                    loss = criterion(output.reshape(-1), capacity.reshape(-1))
                    test_loss += loss.item() * data.shape[0]
            print(
                'Epoch [{}/{}], Test Loss: {:.4f}'.format(epoch + 1, args.num_epochs, test_loss / len(test_dataset)))

    else:
        raise NotImplementedError


