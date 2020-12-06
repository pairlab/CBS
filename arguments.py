import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--log_name', type=str, default='cbs')
    parser.add_argument('--alg', type=str, default='res', choices=['normal', 'vgg', 'res', 'wrn'])
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--ssl', action='store_true')
    parser.add_argument('--percentage', type=int, default=10)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-1)

    # CBS ARGS
    parser.add_argument('--std', default=1, type=float)
    parser.add_argument('--std_factor', default=0.9, type=float)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--kernel_size', default=3, type=int)

    args = parser.parse_args()

    args.cuda = True if torch.cuda.is_available() and not args.no_cuda else False

    return args

