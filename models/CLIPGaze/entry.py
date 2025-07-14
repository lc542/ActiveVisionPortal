import argparse
from models.CLIPGaze.model import test as eval_module, train as train_module
from models.CLIPGaze.model.utils import get_args_parser_train, get_args_parser_test


def train(unknown_args):
    parser = get_args_parser_train()
    args = parser.parse_args(unknown_args)
    train_module.main(args)


def eval(unknown_args):
    parser = get_args_parser_test()
    args = parser.parse_args(unknown_args)
    eval_module.main(args)


def help():
    print("=== CLIPGaze Train Arguments ===")
    get_args_parser_train().print_help()

    print("\n=== CLIPGaze Eval Arguments ===")
    get_args_parser_test().print_help()
