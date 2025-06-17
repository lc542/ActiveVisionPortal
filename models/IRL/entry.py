import argparse
import torch
import numpy as np
import os
import sys

from models.IRL.model import train as train_module
from models.IRL.model import eval as eval_module
from models.IRL.model.config import JsonConfig

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))


def get_args_parser_train():
    parser = argparse.ArgumentParser(description='IRL Training Arguments')
    parser.add_argument('--hparams', type=str, default=os.path.join(ROOT, 'hparams/coco_search18.json'),
                        help='Path to hparams JSON file')
    parser.add_argument('--dataset_root', type=str, default=os.path.join(ROOT, 'data/trainval'),
                        help='Root path to training dataset')
    parser.add_argument('--annotation_root', type=str, default=os.path.join(PROJECT_ROOT, 'datasets/COCO-Search18'),
                        help='Root path to annotation')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device id')
    return parser


def get_args_parser_eval():
    parser = argparse.ArgumentParser(description='IRL Evaluation Arguments')
    parser.add_argument('--hparams', type=str, default=os.path.join(ROOT, 'hparams/coco_search18.json'),
                        help='Path to hparams JSON file')
    parser.add_argument('--dataset_root', type=str, default=os.path.join(ROOT, 'data/trainval'),
                        help='Root path to training dataset')
    parser.add_argument('--test_dataset_root', type=str, default=os.path.join(ROOT, 'data/test'),
                        help='Root path to test dataset')
    parser.add_argument('--annotation_root', type=str, default=os.path.join(PROJECT_ROOT, 'datasets/COCO-Search18'),
                        help='Root path to annotation')
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(ROOT, 'pretrained_models'),
                        help='Path to trained generator .pkg')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device id')
    return parser


def train(unknown_args):
    parser = get_args_parser_train()
    args = parser.parse_args(unknown_args)
    hparams = JsonConfig(args.hparams)

    # 切换 device
    device = torch.device(f'cuda:{args.cuda}')
    torch.manual_seed(42619)
    np.random.seed(42619)

    train_module.main(hparams, args.dataset_root, device, args.annotation_root)


def eval(unknown_args):
    parser = get_args_parser_eval()
    args = parser.parse_args(unknown_args)
    hparams = JsonConfig(args.hparams)

    device = torch.device(f'cuda:{args.cuda}')
    torch.manual_seed(42620)
    np.random.seed(42620)

    eval_module.main(hparams, args.dataset_root, args.test_dataset_root, args.checkpoint_dir, device,
                     args.annotation_root)


def help():
    print("=== IRL Train Arguments ===")
    get_args_parser_train().print_help()
    print("\n=== IRL Eval Arguments ===")
    get_args_parser_eval().print_help()
