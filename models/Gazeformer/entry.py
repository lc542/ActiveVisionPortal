import argparse
from models.Gazeformer.model import eval as eval_module, train as train_module


def get_args_parser_test():
    parser = argparse.ArgumentParser('Gaze Transformer Tester', add_help=False)
    parser.add_argument('--img_ftrs_dir', default='models/Gazeformer/data/image_features', type=str,
                        help="Directory of precomputed ResNet features")
    parser.add_argument('--im_h', default=20, type=int, help="Height of feature map input to encoder")
    parser.add_argument('--im_w', default=32, type=int, help="Width of feature map input to encoder")
    parser.add_argument('--patch_size', default=16, type=int,
                        help="Patch size of feature map input with respect to fixation image dimensions (320X512)")
    parser.add_argument('--max_len', default=7, type=int, help="Maximum length of scanpath")
    parser.add_argument('--num_encoder', default=6, type=int, help="Number of transformer encoder layers")
    parser.add_argument('--num_decoder', default=6, type=int, help="Number of transformer decoder layers")
    parser.add_argument('--hidden_dim', default=512, type=int, help="Hidden dimensionality of transformer layers")
    parser.add_argument('--nhead', default=8, type=int, help="Number of heads for transformer attention layers")
    parser.add_argument('--img_hidden_dim', default=2048, type=int, help="Channel size of initial ResNet feature map")
    parser.add_argument('--lm_hidden_dim', default=768, type=int,
                        help="Dimensionality of target embeddings from language model")
    parser.add_argument('--trained_model', default='models/Gazeformer/checkpoints/gazeformer_cocosearch_TP.pkg',
                        type=str, help="Trained model checkpoint to run for inference")
    parser.add_argument('--seed', default=42, type=int, help="Seed")
    parser.add_argument('--cuda', default=0, type=int, help="CUDA core to load models and data")
    parser.add_argument('--condition', default='present', type=str, help="Search condition (present/absent)")
    parser.add_argument('--zerogaze', default=False, action='store_true', help="ZeroGaze setting flag")
    parser.add_argument('--task', default='car', type=str,
                        help="if evaluation is in ZeroGaze setting, the unseen target to evaluate the model")
    parser.add_argument('--num_samples', default=10, type=int, help="Number of scanpaths sampled per test case")
    return parser


def get_args_parser_train():
    parser = argparse.ArgumentParser('Gaze Transformer Trainer', add_help=False)
    parser.add_argument('--head_lr', default=1e-6, type=float, help="Learning rate for SlowOpt")
    parser.add_argument('--tail_lr', default=1e-4, type=float, help="Learning rate for FastOpt")
    parser.add_argument('--belly_lr', default=2e-6, type=float, help="Learning rate for MidOpt")
    parser.add_argument('--train_file', default='coco_search18_fixations_TP_train.json', type=str,
                        help="Training fixation file")
    parser.add_argument('--valid_file', default='coco_search18_fixations_TP_validation.json', type=str,
                        help="Validation fixation file")
    parser.add_argument('--img_ftrs_dir', default='models/Gazeformer/data/image_features', type=str,
                        help="Directory of precomputed ResNet features")
    parser.add_argument('--im_h', default=20, type=int, help="Height of feature map input to encoder")
    parser.add_argument('--im_w', default=32, type=int, help="Width of feature map input to encoder")
    parser.add_argument('--patch_size', default=16, type=int,
                        help="Patch size of feature map input with respect to fixation image dimensions (320X512)")
    parser.add_argument('--seed', default=42, type=int, help="seed")
    parser.add_argument('--batch_size', default=32, type=int, help="Batch Size")
    parser.add_argument('--epochs', default=200, type=int, help="Maximum number of epochs to train")
    parser.add_argument('--max_len', default=7, type=int, help="Maximum length of scanpath")
    parser.add_argument('--num_encoder', default=6, type=int, help="Number of transformer encoder layers")
    parser.add_argument('--num_decoder', default=6, type=int, help="Number of transformer decoder layers")
    parser.add_argument('--hidden_dim', default=512, type=int, help="Hidden dimensionality of transformer layers")
    parser.add_argument('--nhead', default=8, type=int, help="Number of heads for transformer attention layers")
    parser.add_argument('--img_hidden_dim', default=2048, type=int, help="Channel size of initial ResNet feature map")
    parser.add_argument('--lm_hidden_dim', default=768, type=int,
                        help="Dimensionality of target embeddings from language model")
    parser.add_argument('--encoder_dropout', default=0.1, type=float, help="Encoder dropout rate")
    parser.add_argument('--decoder_dropout', default=0.2, type=float, help="Decoder and fusion step dropout rate")
    parser.add_argument('--cls_dropout', default=0.4, type=float, help="Final scanpath prediction dropout rate")
    parser.add_argument('--retraining', default=False, action='store_true', help="Retraining from a checkpoint")
    parser.add_argument('--last_checkpoint', default='./saved_models/gazeformer_6E_6D_32_512d_70.pkg', type=str,
                        help="Checkpoint for retraining")
    parser.add_argument('--model_root', default='./saved_models/trained', type=str, help="Checkpoint directory")
    parser.add_argument('--cuda', default=0, type=int, help="CUDA core to load models and data")
    parser.add_argument('--num_workers', default=6, type=int, help="Number of workers for data loader")
    return parser


def train(unknown_args, dataset=None):
    # parser.add_argument('--dataset_dir', default='datasets/COCO-Search18', type=str, help="Dataset Directory")
    print("\n=== Gazeformer Train ===")
    parser = get_args_parser_train()
    args = parser.parse_args(unknown_args)
    args.dataset_dir = dataset
    train_module.main(args)


def eval(unknown_args, dataset=None):
    print("\n=== Gazeformer Evaluation ===")
    parser = get_args_parser_test()
    args = parser.parse_args(unknown_args)
    args.dataset_dir = dataset
    eval_module.main(args)


def help():
    print("=== Gazeformer Train Arguments ===")
    get_args_parser_train().print_help()

    print("\n=== Gazeformer Eval Arguments ===")
    get_args_parser_test().print_help()
