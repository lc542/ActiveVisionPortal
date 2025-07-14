import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.abspath(sys.argv[0]))
sys.path.append(ROOT)

from models.Scanpaths.model import train as train_module, test as test_module
from models.Scanpaths.model.opts import parse_opt


def get_args_parser_train():
    return parse_opt()


def get_args_parser_eval():
    parser = argparse.ArgumentParser(description="Scanpath prediction for images")
    parser.add_argument("--mode", type=str, default="validation", help="Selecting running mode (default: validation)")
    parser.add_argument("--img_dir", type=str, default="datasets/COCO-Search18/images", help="Directory to the image data (stimuli)")
    parser.add_argument("--fix_dir", type=str, default="datasets/COCO-Search18", help="Directory to the raw fixation file")
    parser.add_argument("--detector_dir", type=str, default="models/Scanpaths/detectors", help="Directory to the saliency maps")
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
    parser.add_argument("--ablate_attention_info", type=bool, default=False, help="Ablate the attention information or not")
    return parser.parse_args()


def train(unknown_args):
    sys.argv = [sys.argv[0]] + unknown_args
    args = parse_opt(unknown_args)
    args.mode = "train"
    train_module.main(args)


def eval(unknown_args):
    sys.argv = [sys.argv[0]] + unknown_args
    args = get_args_parser_eval()
    args.mode = "validation"
    test_module.main(args)


def help():
    print("=== Scanpaths Train Arguments ===")
    get_args_parser_train().print_help()
    print("\n=== Scanpaths Eval Arguments ===")
    get_args_parser_eval().print_help()


if __name__ == '__main__':
    test_args = [
        "--img_dir", "datasets/COCO-Search18/images",
        "--fix_dir", "datasets/COCO-Search18",
        "--evaluation_dir", "models/Scanpaths/checkpoints"
    ]
    eval(test_args)
