import argparse
import os
import sys

from models.HAT.model import train as train_module
from models.HAT.model.common.config import JsonConfig

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))


def get_args_parser_train():
    parser = argparse.ArgumentParser(description='HAT Training Arguments')
    parser.add_argument('--hparams', type=str, default=os.path.join(ROOT, 'configs/coco_search18_dense_SSL_TP.json'),
                        help='Path to hparams JSON file')
    parser.add_argument('--dataset-root', type=str, default=os.path.join(PROJECT_ROOT, 'datasets/COCO-Search18'),
                        help='Root path to dataset')
    parser.add_argument('--pretrain', action='store_true', help='perform pretraining')
    parser.add_argument('--model', choices=['HAT', 'FOM', 'HATv2'], default='HAT', help='model type')
    parser.add_argument('--transfer-learn', choices=['none', 'search2freeview', 'freeview2search', 'finetune'], default='none', help='setting for transfer learning')
    parser.add_argument('--split', type=int, default=1, help='dataset split for MIT1003/CAT2000 only (default=1)')
    parser.add_argument('--eval-mode', choices=['greedy', 'sample'], type=str, default='greedy', help='whether to sample scanapth or greedily predict scanpath during evaluation (default=greedy)')
    parser.add_argument('--disable-saliency', action='store_true', help='do not calculate saliency metrics')
    parser.add_argument('--gpu-id', type=int, default=0, help='gpu id (default=0)')
    parser.add_argument('--eval-only', action='store_true', help=argparse.SUPPRESS)
    return parser


def get_args_parser_eval():
    parser = argparse.ArgumentParser(description='HAT Evaluation Arguments')
    parser.add_argument('--hparams', type=str, default=os.path.join(ROOT, 'configs/coco_search18_dense_SSL_TP.json'),
                        help='Path to hparams JSON file')
    parser.add_argument('--dataset-root', type=str, default=os.path.join(PROJECT_ROOT, 'datasets/COCO-Search18'),
                        help='Root path to dataset')
    parser.add_argument('--pretrain', action='store_true', help='perform pretraining')
    parser.add_argument('--model', choices=['HAT', 'FOM', 'HATv2'], default='HAT', help='model type')
    parser.add_argument('--transfer-learn', choices=['none', 'search2freeview', 'freeview2search', 'finetune'], default='none', help='setting for transfer learning')
    parser.add_argument('--split', type=int, default=1, help='dataset split for MIT1003/CAT2000 only (default=1)')
    parser.add_argument('--eval-mode', choices=['greedy', 'sample'], type=str, default='greedy', help='whether to sample scanapth or greedily predict scanpath during evaluation (default=greedy)')
    parser.add_argument('--disable-saliency', action='store_true', help='do not calculate saliency metrics')
    parser.add_argument('--gpu-id', type=int, default=0, help='gpu id (default=0)')
    parser.add_argument('--eval-only', action='store_true', help='perform evaluation only')
    return parser


def train(unknown_args):
    parser = get_args_parser_train()
    filtered_args = [arg for arg in unknown_args if arg != '--eval-only']
    args = parser.parse_args(filtered_args)
    args.eval_only = False
    train_module.main(args)


def eval(unknown_args):
    parser = get_args_parser_eval()
    args = parser.parse_args(unknown_args + ['--eval-only'])
    train_module.main(args)


def help():
    print("=== HAT Train Arguments ===")
    get_args_parser_train().print_help()
    print("\n=== HAT Eval Arguments ===")
    get_args_parser_eval().print_help()


if __name__ == '__main__':

    test_args = [
        '--hparams', 'configs/coco_search18_dense_SSL_TP.json',
        '--dataset-root', 'datasets/COCO-Search18'
    ]

    eval(test_args)
