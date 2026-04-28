import json
from datetime import datetime
import torch.nn as nn

from args import get_parser
from utils import *
from mtad_gat import MTAD_GAT
from prediction import Predictor
from training import Trainer
from tqdm import tqdm

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    id = datetime.now().strftime("%d%m%Y_%H%M%S") + f"fold_{args.fold_num}"

    dataset = args.dataset
    window_size = args.lookback
    spec_res = args.spec_res
    normalize = args.normalize
    n_epochs = args.epochs
    batch_size = args.bs
    init_lr = args.init_lr
    val_split = args.val_split
    shuffle_dataset = args.shuffle_dataset
    use_cuda = args.use_cuda
    print_every = args.print_every
    log_tensorboard = args.log_tensorboard
    group_index = args.group[0]
    index = args.group[2:]
    args_summary = str(args.__dict__)
    print(args_summary)

    if dataset == 'SMD':
        output_path = f'output/SMD/{args.group}'
        (x_train, _), (x_test, y_test) = get_data(f"machine-{group_index}-{index}", normalize=normalize)
    elif dataset in ['MSL', 'SMAP']:
        output_path = f'output/{dataset}'
        (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    elif dataset in ["BATTERY_BRAND1", "BATTERY_BRAND2", "BATTERY_BRAND3", "BATTERY_BRAND123"]:
        output_path = f'output/{dataset}'
        # (x_train, _), (x_test, y_test) = get_data(dataset, normalize=normalize)
    else:
        raise Exception(f'Dataset "{dataset}" not available.')

    log_dir = f'{output_path}/logs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    save_path = f"{output_path}/{id}"

    # x_train = torch.from_numpy(x_train).float()
    # x_test = torch.from_numpy(x_test).float()
    # n_features = x_train.shape[1]
    n_features = 6

    target_dims = get_target_dims(dataset)
    if target_dims is None:
        out_dim = n_features
        print(f"Will forecast and reconstruct all {n_features} input features")
    elif type(target_dims) == int:
        print(f"Will forecast and reconstruct input feature: {target_dims}")
        out_dim = 1
    else:
        print(f"Will forecast and reconstruct input features: {target_dims}")
        out_dim = len(target_dims)

    if args.battery_brand1:
        train_dataset = SlidingWindowDataset_battery_fivefold_brand1(window_size, target_dims, fold_num=args.fold_num, train=True)
        test_dataset = SlidingWindowDataset_battery_fivefold_brand1(window_size, target_dims, fold_num=args.fold_num, train=False)
    elif args.battery_brand2:
        train_dataset = SlidingWindowDataset_battery_fivefold_brand2(window_size, target_dims, fold_num=args.fold_num, train=True)
        test_dataset = SlidingWindowDataset_battery_fivefold_brand2(window_size, target_dims, fold_num=args.fold_num, train=False)
    elif args.battery_brand3:
        train_dataset = SlidingWindowDataset_battery_fivefold_brand3(window_size, target_dims, fold_num=args.fold_num, train=True)
        test_dataset = SlidingWindowDataset_battery_fivefold_brand3(window_size, target_dims, fold_num=args.fold_num, train=False)
    elif args.battery_brand123:
            train_dataset = SlidingWindowDataset_battery_fivefold_brand123(window_size, target_dims, fold_num=args.fold_num, train=True)
            test_dataset = SlidingWindowDataset_battery_fivefold_brand123(window_size, target_dims, fold_num=args.fold_num, train=False)
    else:
        raise ValueError
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, batch_size, val_split=0, shuffle=shuffle_dataset, test_dataset=test_dataset
    )

    model = MTAD_GAT(
        n_features,
        window_size,
        out_dim,
        kernel_size=args.kernel_size,
        use_gatv2=args.use_gatv2,
        feat_gat_embed_dim=args.feat_gat_embed_dim,
        time_gat_embed_dim=args.time_gat_embed_dim,
        gru_n_layers=args.gru_n_layers,
        gru_hid_dim=args.gru_hid_dim,
        forecast_n_layers=args.fc_n_layers,
        forecast_hid_dim=args.fc_hid_dim,
        recon_n_layers=args.recon_n_layers,
        recon_hid_dim=args.recon_hid_dim,
        dropout=args.dropout,
        alpha=args.alpha
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    forecast_criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    trainer = Trainer(
        model,
        optimizer,
        window_size,
        n_features,
        target_dims,
        n_epochs,
        batch_size,
        init_lr,
        forecast_criterion,
        recon_criterion,
        use_cuda,
        save_path,
        log_dir,
        print_every,
        log_tensorboard,
        args_summary
    )

    trainer.fit(train_loader, val_loader)

    plot_losses(trainer.losses, save_path=save_path, plot=False)

    # Check test loss
    test_loss = trainer.evaluate(test_loader)
    print(f"Test forecast loss: {test_loss[0]:.5f}")
    print(f"Test reconstruction loss: {test_loss[1]:.5f}")
    print(f"Test total loss: {test_loss[2]:.5f}")

    # Some suggestions for POT args
    level_q_dict = {
        "SMAP": (0.90, 0.005),
        "MSL": (0.90, 0.001),
        "SMD-1": (0.9950, 0.001),
        "SMD-2": (0.9925, 0.001),
        "SMD-3": (0.9999, 0.001)
    }
    # key = "SMD-" + args.group[0] if args.dataset == "SMD" else args.dataset
    # level, q = level_q_dict[key]
    # if args.level is not None:
    #     level = args.level
    # if args.q is not None:
    #     q = args.q
    #
    # # Some suggestions for Epsilon args
    # reg_level_dict = {"SMAP": 0, "MSL": 0, "SMD-1": 1, "SMD-2": 1, "SMD-3": 1}
    # key = "SMD-" + args.group[0] if dataset == "SMD" else dataset
    # reg_level = reg_level_dict[key]

    trainer.load(f"{save_path}/model.pt")
    prediction_args = {
        'dataset': dataset,
        "target_dims": target_dims,
        'scale_scores': args.scale_scores,
        # "level": level,
        # "q": q,
        'dynamic_pot': args.dynamic_pot,
        "use_mov_av": args.use_mov_av,
        "gamma": args.gamma,
        # "reg_level": reg_level,
        "save_path": save_path,
    }
    best_model = trainer.model
    # predictor = Predictor(
    #     best_model,
    #     window_size,
    #     n_features,
    #     prediction_args,
    # )

    # label = y_test[window_size:] if y_test is not None else None
    # predictor.predict_anomalies(x_train, x_test, label)
    
    # get score
    best_model.eval()
    preds = []
    recons = []
    values = []
    car_nums = []
    car_charge_segment = []
    car_head = []
    use_cuda = True
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        for x, y, carnum, charge_segment, head in tqdm(train_loader):
            x = x.to(device)
            y = y.to(device)

            y_hat, _ = best_model(x)

            # Shifting input to include the observed value (y) when doing the reconstruction
            recon_x = torch.cat((x[:, 1:, :], y), dim=1)
            _, window_recon = best_model(recon_x)

            preds.append(y_hat.detach().cpu().numpy())
            # Extract last reconstruction only
            recons.append(window_recon[:, -1, :].detach().cpu().numpy())
            values.append(y.detach().cpu().numpy())

            car_nums.append(carnum)
            car_charge_segment.append(charge_segment)
            car_head.append(head)

    with torch.no_grad():
        for x, y, carnum, charge_segment, head in tqdm(test_loader):
            x = x.to(device)
            y = y.to(device)

            y_hat, _ = best_model(x)

            # Shifting input to include the observed value (y) when doing the reconstruction
            recon_x = torch.cat((x[:, 1:, :], y), dim=1)
            _, window_recon = best_model(recon_x)

            preds.append(y_hat.detach().cpu().numpy())
            # Extract last reconstruction only
            recons.append(window_recon[:, -1, :].detach().cpu().numpy())
            values.append(y.detach().cpu().numpy())
            car_nums.append(carnum)
            car_charge_segment.append(charge_segment)
            car_head.append(head)

    preds = np.concatenate(preds, axis=0)
    recons = np.concatenate(recons, axis=0)
    actual = np.squeeze(np.concatenate(values, axis=0))
    car_nums = np.squeeze(np.concatenate(car_nums, axis=0))
    car_charge_segment = np.squeeze(np.concatenate(car_charge_segment, axis=0))
    car_head = np.squeeze(np.concatenate(car_head, axis=0))

    np.save(f"{save_path}/preds.txt", preds)
    np.save(f"{save_path}/recons.txt", recons)
    np.save(f"{save_path}/actual.txt", actual)
    np.save(f"{save_path}/car_nums.txt", car_nums)
    np.save(f"{save_path}/car_charge_segment.txt", car_charge_segment)
    np.save(f"{save_path}/car_head.txt", car_head)


    # Save config
    args_path = f"{save_path}/config.txt"
    with open(args_path, "w") as f:
        json.dump(args.__dict__, f, indent=2)

'''
CUDA_VISIBLE_DEVICES=0 python train.py --dataset battery_brand1 --battery_brand1 --fold_num 0 --use_gatv2 False --epochs 30 --lookback 127
CUDA_VISIBLE_DEVICES=1 python train.py --dataset battery_brand1 --battery_brand1 --fold_num 1 --use_gatv2 False --epochs 30 --lookback 127
CUDA_VISIBLE_DEVICES=2 python train.py --dataset battery_brand1 --battery_brand1 --fold_num 2 --use_gatv2 False --epochs 30 --lookback 127
CUDA_VISIBLE_DEVICES=3 python train.py --dataset battery_brand1 --battery_brand1 --fold_num 3 --use_gatv2 False --epochs 30 --lookback 127
CUDA_VISIBLE_DEVICES=0 python train.py --dataset battery_brand1 --battery_brand1 --fold_num 4 --use_gatv2 False --epochs 30 --lookback 127

CUDA_VISIBLE_DEVICES=0 python train.py --dataset battery_brand123 --battery_brand123 --fold_num 0 --use_gatv2 False --epochs 30 --lookback 127
CUDA_VISIBLE_DEVICES=1 python train.py --dataset battery_brand123 --battery_brand123 --fold_num 1 --use_gatv2 False --epochs 30 --lookback 127
CUDA_VISIBLE_DEVICES=2 python train.py --dataset battery_brand123 --battery_brand123 --fold_num 2 --use_gatv2 False --epochs 30 --lookback 127
CUDA_VISIBLE_DEVICES=3 python train.py --dataset battery_brand123 --battery_brand123 --fold_num 3 --use_gatv2 False --epochs 30 --lookback 127
CUDA_VISIBLE_DEVICES=1 python train.py --dataset battery_brand123 --battery_brand123 --fold_num 4 --use_gatv2 False --epochs 30 --lookback 127

CUDA_VISIBLE_DEVICES=0 python train.py --dataset battery_brand2 --battery_brand2 --fold_num 0 --use_gatv2 False --epochs 30 --lookback 127
CUDA_VISIBLE_DEVICES=1 python train.py --dataset battery_brand2 --battery_brand2 --fold_num 1 --use_gatv2 False --epochs 30 --lookback 127
CUDA_VISIBLE_DEVICES=2 python train.py --dataset battery_brand2 --battery_brand2 --fold_num 2 --use_gatv2 False --epochs 30 --lookback 127
CUDA_VISIBLE_DEVICES=2 python train.py --dataset battery_brand2 --battery_brand2 --fold_num 3 --use_gatv2 False --epochs 30 --lookback 127
CUDA_VISIBLE_DEVICES=2 python train.py --dataset battery_brand2 --battery_brand2 --fold_num 4 --use_gatv2 False --epochs 30 --lookback 127

'''