import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(ROOT)

from models.Scanpaths.model import train as train_module, test as test_module
from models.Scanpaths.model.utils.config import CfgNode

def get_args_parser_train():
    parser = argparse.ArgumentParser(description="Scanpath prediction for images")
    parser.add_argument("--mode", type=str, default="train", help="Selecting running mode (default: train)")
    parser.add_argument("--detector_dir", type=str, default="models/Scanpaths/detectors",
                        help="Directory to detector results")
    parser.add_argument("--width", type=int, default=320, help="Width of input data")
    parser.add_argument("--height", type=int, default=240, help="Height of input data")
    parser.add_argument("--map_width", type=int, default=40, help="Height of output data")
    parser.add_argument("--map_height", type=int, default=30, help="Height of output data")
    parser.add_argument("--blur_sigma", type=float, default=None, help="Standard deviation for Gaussian kernel")
    parser.add_argument("--detector_threshold", type=float, default=0.8, help="threshold for the detector")
    parser.add_argument("--clip", type=float, default=12.5, help="Gradient clipping")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs")
    parser.add_argument("--warmup_epoch", type=int, default=1, help="Epoch when finishing warn up strategy")
    parser.add_argument("--start_rl_epoch", type=int, default=5, help="Epoch when starting reinforcement learning")
    parser.add_argument("--rl_sample_number", type=int, default=5,
                        help="Number of samples used in policy gradient update")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--rl_lr_initial_decay", type=float, default=0.5, help="Initial decay of learning rate of rl")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='List of GPU ids')
    parser.add_argument("--log_root", type=str, default="./assets/", help="Log root")
    parser.add_argument("--resume_dir", type=str, default="", help="Resume from a specific directory")
    parser.add_argument("--center_bias", type=bool, default=True, help="Adding center bias or not")
    parser.add_argument("--lambda_1", type=float, default=1, help="Hyper-parameter for duration loss term")
    parser.add_argument("--eval_repeat_num", type=int, default=10, help="Repeat number for evaluation")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
    parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")
    parser.add_argument("--ablate_attention_info", type=bool, default=False,
                        help="Ablate the attention information or not")
    parser.add_argument("--supervised_save", type=bool, default=True,
                        help="Copy the files before start the policy gradient update")

    # config
    parser.add_argument('--cfg', type=str, default=None,
                        help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]\n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')

    return parser


def get_args_parser_eval():
    parser = argparse.ArgumentParser(description="Scanpath prediction for images")
    parser.add_argument("--mode", type=str, default="test", help="Selecting running mode")
    parser.add_argument("--detector_dir", type=str, default="models/Scanpaths/detectors",
                        help="Directory to the saliency maps")
    parser.add_argument("--width", type=int, default=320, help="Width of input data")
    parser.add_argument("--height", type=int, default=240, help="Height of input data")
    parser.add_argument("--map_width", type=int, default=40, help="Height of output data")
    parser.add_argument("--map_height", type=int, default=30, help="Height of output data")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--detector_threshold", type=float, default=0.8, help="threshold for the detector")
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='List of GPU ids')
    parser.add_argument("--evaluation_dir", type=str, default="models/Scanpaths/checkpoints",
                        help="Resume from a specific directory")
    parser.add_argument("--eval_repeat_num", type=int, default=10, help="Repeat number for evaluation")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the generated scanpath")
    parser.add_argument("--max_length", type=int, default=16, help="Maximum length of the generated scanpath")
    parser.add_argument("--ablate_attention_info", type=bool, default=False,
                        help="Ablate the attention information or not")
    return parser


def train(unknown_args, dataset=None):
    print("\n=== Scanpaths Train ===")
    parser = get_args_parser_train()
    args = parser.parse_args(unknown_args)
    if args.cfg is not None or args.set_cfgs is not None:
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k, v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' % k)
            setattr(args, k, v)
        args = parser.parse_args(unknown_args, namespace=args)

    args.fix_dir = dataset
    args.img_dir = os.path.join(args.fix_dir, "images")
    train_module.main(args)


def eval(unknown_args, dataset=None):
    print("\n=== Scanpaths Evaluation ===")
    parser = get_args_parser_eval()
    args = parser.parse_args(unknown_args)
    print(args)
    args.fix_dir = dataset
    args.img_dir = os.path.join(args.fix_dir, "images")
    test_module.main(args)


def help():
    print("=== Scanpaths Train Arguments ===")
    get_args_parser_train().print_help()
    print("\n=== Scanpaths Eval Arguments ===")
    get_args_parser_eval().print_help()
