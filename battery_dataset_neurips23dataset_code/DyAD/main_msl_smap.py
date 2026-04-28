import argparse
import json
import os
import sys
import train
import extract
import evaluate

import torch
import random
import numpy as np

seed = 2


if __name__ == '__main__':
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='PyTorch on MSL/SMAP Example')
    parser.add_argument('--config_path', type=str, default='./params.json')

    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=p_args)
    print("Loaded configs at %s" % args.config_path)
    print("args", args)

    reconstruct=False

    # train and save feature
    tr = train.Train_spacecraft(args)
    print('train start............................')
    tr.main(reconstruct=reconstruct)
    print('train end............................')
    modelparams_path=tr.getmodelparams()
    del tr
    parser.add_argument('--modelparams_path', type=str,
                        default=modelparams_path)
    args = parser.parse_args()
    with open(args.modelparams_path, 'r') as file:
        p_args = argparse.Namespace()
        model_params=json.load(file)
        p_args.__dict__.update(model_params["args"])
        args = parser.parse_args(namespace=p_args)
    # feature extraction
    ext = extract.Extraction_spacecraft(args)
    print('feature extraction start............................')
    ext.main(reconstruct=reconstruct)
    print('feature extraction end............................')
    del ext



'''
CUDA_VISIBLE_DEVICES=0 python main_msl_smap.py --config_path params_smap.json
'''
