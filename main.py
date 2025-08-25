import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import sys
from model_registry import MODEL_REGISTRY


# train
# python main.py --model irl --train

# eval
# python main.py --model gazeformer --eval

# help
# python main.py --model irl --help_model

# list
# python main.py --list_models

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Framework Entry")
    parser.add_argument('--model', type=str,
                        help='Model name to run (e.g., irl, gazeformer)')
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--eval', action='store_true', help='Run evaluation')
    parser.add_argument('--dataset', type=str, default="datasets/COCO-Search18", help='Dataset Directory')
    parser.add_argument('--help_model', action='store_true', help='Show model-specific help message')
    parser.add_argument('--list_models', action='store_true', help='List all available models')

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


def main():
    args, unknown_args = parse_args()
    # print(f"Main args: {args}")
    # print(f"Model-specific args: {unknown_args}")

    if args.list_models:
        print("Available models:")
        for model_name in MODEL_REGISTRY:
            print(f"  - {model_name}")
        return

    if not args.model:
        print("[Error] --model is required unless using --help_models")
        return

    model_name = args.model.lower()
    if model_name not in MODEL_REGISTRY:
        print(f"[Error] Unknown model: {model_name}")
        print(f"[Available models] {list(MODEL_REGISTRY.keys())}")
        return

    entry_module = MODEL_REGISTRY[model_name]

    if args.train:
        entry_module.train(unknown_args, dataset=args.dataset)
    elif args.eval:
        entry_module.eval(unknown_args, dataset=args.dataset)
    elif args.help_model:
        entry_module.help()
    else:
        print("[Error] Please specify one of --train, --eval, --help_model or --list_models")


if __name__ == '__main__':
    main()
