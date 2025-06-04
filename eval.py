"""Test script with evaluation.
Usage:
  test.py <hparams> <checkpoint_dir> <dataset_root> [--cuda=<id>]
  test.py -h | --help

Options:
  -h --help     Show this screen.
  --cuda=<id>   id of the cuda device [default: 0].
"""

import os
import json
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from docopt import docopt
from os.path import join
from models.IRL.dataset import process_data
from models.IRL.irl_dcb.config import JsonConfig
import cv2 as cv

from models.IRL.irl_dcb.data import LHF_IRL
from models.IRL.irl_dcb.models import LHF_Policy_Cond_Small
from models.IRL.irl_dcb.environment import IRL_Env4LHF
from models.IRL.irl_dcb import utils, metrics
from models.IRL.irl_dcb.utils import compute_search_cdf

torch.manual_seed(42620)
np.random.seed(42620)


def gen_scanpaths(generator,
                  env_test,
                  test_img_loader,
                  patch_num,
                  max_traj_len,
                  im_w,
                  im_h,
                  num_sample=10):
    all_actions = []
    for i_sample in range(num_sample):
        progress = tqdm(test_img_loader,
                        desc='trial ({}/{})'.format(i_sample + 1, num_sample))
        for i_batch, batch in enumerate(progress):
            env_test.set_data(batch)
            img_names_batch = batch['img_name']
            cat_names_batch = batch['cat_name']
            with torch.no_grad():
                env_test.reset()
                trajs = utils.collect_trajs(env_test,
                                            generator,
                                            patch_num,
                                            max_traj_len,
                                            is_eval=True,
                                            sample_action=True)
                all_actions.extend([(cat_names_batch[i], img_names_batch[i],
                                     'present', trajs['actions'][:, i])
                                    for i in range(env_test.batch_size)])

    scanpaths = utils.actions2scanpaths(all_actions, patch_num, im_w, im_h)
    utils.cutFixOnTarget(scanpaths, bbox_annos)

    return scanpaths


if __name__ == '__main__':
    # python models/IRL/my_eval.py models/IRL/hparams/coco_search18.json models/IRL/trained_models datasets/COCO-Search18/IRL --cuda=0
    args = docopt(__doc__)
    device = torch.device('cuda:{}'.format(args['--cuda']))
    hparams = JsonConfig(args["<hparams>"])
    checkpoint = args["<checkpoint_dir>"]
    dataset_root = args["<dataset_root>"]

    DCB_dir_HR = join(dataset_root, 'DCBs/HR/')
    DCB_dir_LR = join(dataset_root, 'DCBs/LR/')

    DCB_dir_HR_test = join(dataset_root, 'test', 'DCBs/HR/')
    DCB_dir_LR_test = join(dataset_root, 'test', 'DCBs/LR/')

    data_name = '{}x{}'.format(hparams.Data.im_w, hparams.Data.im_h)

    bbox_annos = np.load(join(dataset_root, 'bbox_annos.npy'), allow_pickle=True).item()
    bbox_annos_test = np.load(join(dataset_root, 'test', 'bbox_annos.npy'), allow_pickle=True).item()

    with open(join(dataset_root, 'human_scanpaths_TP_trainval_train.json')) as f:
        human_scanpaths_train = json.load(f)
    with open(join(dataset_root, 'human_scanpaths_TP_trainval_valid.json')) as f:
        human_scanpaths_valid = json.load(f)
    with open(join(dataset_root, 'test', 'human_scanpaths_TP_test_rescaled.json')) as f:
        human_gt = json.load(f)

    all_init_fix_trajs = human_scanpaths_train + human_scanpaths_valid + human_gt

    target_init_fixs = {
        traj['task'] + '_' + traj['name']: (traj['X'][0] / hparams.Data.im_w,
                                            traj['Y'][0] / hparams.Data.im_h)
        for traj in all_init_fix_trajs if len(traj['X']) > 0
    }

    cat_names = list(np.unique([x['task'] for x in human_scanpaths_train]))
    catIds = dict(zip(cat_names, list(range(len(cat_names)))))

    dataset = process_data(human_scanpaths_train, human_scanpaths_valid,
                           DCB_dir_HR, DCB_dir_LR, bbox_annos, hparams)

    train_task_img_pair = np.unique([
        traj['task'] + '_' + traj['name'] for traj in human_scanpaths_train
    ])

    test_task_img_pair = np.unique([traj['task'] + '_' + traj['name'] for traj in human_gt])

    test_dataset = LHF_IRL(DCB_dir_HR_test, DCB_dir_LR_test, target_init_fixs,
                           test_task_img_pair, bbox_annos_test,
                           hparams.Data, catIds)

    # test_dataset = LHF_IRL(DCB_dir_HR, DCB_dir_LR, target_init_fixs,
    #                        train_task_img_pair, bbox_annos,
    #                        hparams.Data, catIds)

    dataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=16,
                                             shuffle=False,
                                             num_workers=2)

    input_size = 134
    task_eye = torch.eye(len(dataset['catIds'])).to(device)
    generator = LHF_Policy_Cond_Small(hparams.Data.patch_count,
                                      len(dataset['catIds']), task_eye,
                                      input_size).to(device)

    generator.load_state_dict(torch.load(join(checkpoint, 'trained_generator.pkg'),
                                         map_location=device)["model"])
    generator.eval()

    env_test = IRL_Env4LHF(hparams.Data,
                           max_step=hparams.Data.max_traj_length,
                           mask_size=hparams.Data.IOR_size,
                           status_update_mtd=hparams.Train.stop_criteria,
                           device=device,
                           inhibit_return=True)

    print('sample scanpaths (10 for each testing image)...')
    predictions = gen_scanpaths(generator,
                                env_test,
                                dataloader,
                                hparams.Data.patch_num,
                                hparams.Data.max_traj_length,
                                hparams.Data.im_w,
                                hparams.Data.im_h,
                                num_sample=1)

    # Evaluation
    # MultiMatch
    mm_score = metrics.compute_mm(human_gt, predictions, hparams.Data.im_w, hparams.Data.im_h)
    print("MultiMatch (Shape, Length, Position, Direction):", mm_score)

    # Scanpath Efficiency
    avg_sp_ratio = metrics.compute_avgSPRatio(predictions, bbox_annos_test, hparams.Data.max_traj_length)
    print("Avg Scanpath Ratio:", avg_sp_ratio)


    # Debug check for ID match
    def get_key(s):
        if 'img' in s:
            return s['img']
        elif 'task' in s and 'name' in s:
            return s['task'] + '_' + s['name'].replace('.jpg', '')
        else:
            return None


    pred_keys = set(get_key(s) for s in predictions if get_key(s) is not None)
    gt_keys = set(get_key(s) for s in human_gt if get_key(s) is not None)
    anno_keys = set(k.replace('.jpg', '') for k in bbox_annos_test.keys())

    print(f"[Debug] Prediction - Anno matched: {len(pred_keys & anno_keys)} / {len(pred_keys)}")
    print(f"[Debug] GT        - Anno matched: {len(gt_keys & anno_keys)} / {len(gt_keys)}")

    filtered_gt = [s for s in human_gt if s.get('fixOnTarget', False)]
    print(f"[Debug]GT : {len(filtered_gt)} / {len(human_gt)}")

    human_mean_cdf, human_scanpaths_grouped = compute_search_cdf(filtered_gt, bbox_annos_test,
                                                                 max_step=hparams.Data.max_traj_length)
    model_cdf, _ = compute_search_cdf(predictions, bbox_annos_test,
                                      max_step=hparams.Data.max_traj_length)

    # TFP-AUC
    auc = metrics.compute_cdf_auc(model_cdf)
    print("TFP-AUC:", auc)

    # Probability mismatch
    prob_mismatch = metrics.compute_prob_mismatch(model_cdf, human_mean_cdf)
    print("Probability Mismatch:", prob_mismatch)

    # # clusters.pkl（ Sequence Score）
    # with open("test_clusters.pkl", "rb") as f:
    #     clusters = pickle.load(f)

    # # Sequence Score
    # seq_score = metrics.get_seq_score(predictions, clusters, hparams.Data.max_traj_length)
    # print("Sequence Score:", seq_score)
