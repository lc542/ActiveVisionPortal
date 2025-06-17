import argparse
from models.Gazeformer.model import eval as eval_module, train as train_module
from models.Gazeformer.model.utils import get_args_parser_train, get_args_parser_test


def train(unknown_args):
    parser = get_args_parser_train()
    args = parser.parse_args(unknown_args)
    train_module.main(args)


def eval(unknown_args):
    parser = get_args_parser_test()
    args = parser.parse_args(unknown_args)
    eval_module.main(args)


def help():
    print("=== Gazeformer Train Arguments ===")
    get_args_parser_train().print_help()

    print("\n=== Gazeformer Eval Arguments ===")
    get_args_parser_test().print_help()
