import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='MS_HaKAN - HSI Multi-Scale Stock Forecasting')

    # Random seed for reproducibility
    parser.add_argument('--random_seed', type=int, default=2021, help='random seed')

    # Basic config
    parser.add_argument('--is_training', type=int, default=1, help='training status')
    parser.add_argument('--model_id', type=str, default='hsi_mshakan_v3_decomp', help='model id')
    parser.add_argument('--model', type=str, default='MS_HaKAN', help='model name: MS_HaKAN')

    # Data loader configuration
    parser.add_argument('--data', type=str, default='custom', help='dataset type: custom')
    parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of data')
    parser.add_argument('--data_path', type=str, default='hsi_custom.csv', help='data file name')
    parser.add_argument('--features', type=str, default='M', help='M: multivariate')
    parser.add_argument('--target', type=str, default='Close', help='target feature')
    parser.add_argument('--freq', type=str, default='d', help='d: daily frequency')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoints')

    # Forecasting window sizes
    parser.add_argument('--seq_len', type=int, default=60, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=30, help='start token length')
    parser.add_argument('--pred_len', type=int, default=30, help='prediction length')

    # Multi-Scale Patching Parameters
    # With decomposition active, a fine/coarse mix can safely isolate trends vs residuals
    parser.add_argument('--patch_len', type=int, default=16, help='patch length 1')
    parser.add_argument('--stride', type=int, default=8, help='stride 1')
    parser.add_argument('--patch_len_2', type=int, default=8, help='patch length 2')
    parser.add_argument('--stride_2', type=int, default=4, help='stride 2')
    
    # Model architecture defaults
    parser.add_argument('--padding_patch', default='end', help='padding type')
    parser.add_argument('--revin', type=int, default=1, help='RevIN')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine')
    parser.add_argument('--subtract_last', type=int, default=0, help='subtract last')
    
    # CRITICAL TRIGGER 1: Activate Series Decomposition to split Trend and Residuals
    parser.add_argument('--decomposition', type=int, default=1, help='decomposition')
    parser.add_argument('--kernel_size', type=int, default=25, help='kernel size for moving avg')
    
    # CRITICAL TRIGGER 2: Re-enable cross-channel fusion (Isolation failed)
    parser.add_argument('--individual', type=int, default=0, help='individual head')
    
    # Regularization
    parser.add_argument('--fc_dropout', type=float, default=0.1, help='fc dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')

    # CRITICAL TRIGGER 3: Shrink model capacity down to choke out overfitting
    parser.add_argument('--embed_type', type=int, default=0, help='embedding type')
    parser.add_argument('--enc_in', type=int, default=8, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=8, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=8, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='SHRUNK: 512 -> 128')
    parser.add_argument('--n_heads', type=int, default=4, help='SHRUNK: 8 -> 4')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='SHRUNK: 2048 -> 512')
    
    parser.add_argument('--moving_avg', type=int, default=25, help='moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false', default=True, help='distilling')
    parser.add_argument('--embed', type=str, default='timeF', help='time features')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='output attention')
    parser.add_argument('--do_predict', action='store_true', help='predict')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='Runway up to 100')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--patience', type=int, default=15, help='Patience set to 15 for smaller model')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--des', type=str, default='mshakan_decomp_shrunk', help='description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type3', help='learning rate adjustment')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', default=False, help='mixed precision')

    # GPU setup
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='multi gpu')
    parser.add_argument('--devices', type=str, default='0', help='device ids')
    parser.add_argument('--test_flop', action='store_true', default=False, help='test flops')

    args = parser.parse_args()

    # Set random seeds
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("="*80)
    print("MS_HaKAN - V3 HAIL MARY: DECOMPOSITION & CAPACITY REGULARIZATION")
    print("="*80)
    print(f"Model: {args.model} | Series Decomposition: ON (Kernel={args.kernel_size})")
    print(f"Model Dimensions: d_model={args.d_model}, d_ff={args.d_ff}, heads={args.n_heads}")
    print(f"Scale 1: Patch={args.patch_len}, Stride={args.stride}")
    print(f"Scale 2: Patch={args.patch_len_2}, Stride={args.stride_2}")
    print(f"Batch Size: {args.batch_size} | Max Epochs: {args.train_epochs}")
    print("="*80)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl1_{}_st1_{}_pl2_{}_st2_{}_dec{}_dm{}_nh{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.patch_len,
                args.stride,
                args.patch_len_2,
                args.stride_2,
                args.decomposition,
                args.d_model,
                args.n_heads,
                args.des, ii)

            print(f"\n[Iteration {ii+1}/{args.itr}]")
            print(f"Setting: {setting}\n")

            exp = Exp(args)
            
            print('>>>>>>>start training : >>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)

            print('\n>>>>>>>testing : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)

            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()