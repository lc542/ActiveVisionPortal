import torch
import numpy as np
import json
from docopt import docopt
import os
from os.path import join

from .config import JsonConfig
from .dataset import process_data
from .builder import build
from .trainer import Trainer


def main(hparams, dataset_root, device, annotation_root):
    torch.manual_seed(42619)
    np.random.seed(42619)
    # args = docopt(__doc__)
    # device = torch.device('cuda:{}'.format(args['--cuda']))
    # # hparams_path = args["<hparams>"]
    # # dataset_root = args["<dataset_root>"]
    # hparams_path = 'models/IRL/hparams/coco_search18.json'
    # dataset_root = 'models/IRL/data/trainval'
    # hparams = JsonConfig(hparams_path)

    # dir of pre-computed beliefs
    DCB_dir_HR = join(dataset_root, 'DCBs/HR/')
    DCB_dir_LR = join(dataset_root, 'DCBs/LR/')

    # bounding box of the target object (for search efficiency evaluation)
    bbox_annos = np.load(join(dataset_root, 'bbox_annos.npy'), allow_pickle=True).item()

    # load ground-truth human scanpaths
    with open(os.path.join(annotation_root, 'coco_search18_fixations_TP_train.json')) as f:
        human_scanpaths_train = json.load(f)
    with open(os.path.join(annotation_root, 'coco_search18_fixations_TP_validation.json')) as f:
        human_scanpaths_valid = json.load(f)

    # exclude incorrect scanpaths
    if hparams.Train.exclude_wrong_trials:
        human_scanpaths_train = list(filter(lambda x: x['correct'] == 1, human_scanpaths_train))
        human_scanpaths_valid = list(filter(lambda x: x['correct'] == 1, human_scanpaths_valid))

    # process fixation data
    dataset = process_data(human_scanpaths_train, human_scanpaths_valid,
                           DCB_dir_HR, DCB_dir_LR, bbox_annos, hparams)

    built = build(hparams, is_training=True, device=device, catIds=dataset['catIds'])
    trainer = Trainer(**built, dataset=dataset, device=device, hparams=hparams)
    trainer.train()


if __name__ == '__main__':
    import multiprocessing

    multiprocessing.set_start_method('spawn', force=True)
    hparams = JsonConfig('models/IRL/hparams/coco_search18.json')
    device = torch.device('cuda:0')
    dataset_root = 'models/IRL/data/trainval'
    annotation_root = 'datasets/COCO-Search18'

    main(hparams, dataset_root, device, annotation_root)
