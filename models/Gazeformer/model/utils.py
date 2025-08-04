import numpy as np
import os
import random
import torch
import argparse
from os.path import join
import scipy.ndimage as filters

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fixations2seq(fixations, max_len):
    processed_fixs = []
    for fix in fixations:
        processed_fixs.append({'tgt_seq_y': torch.tensor(np.array(fix['Y'])[:max_len]),
                               'tgt_seq_x': torch.tensor(np.array(fix['X'])[:max_len]),
                               'tgt_seq_t': torch.tensor(np.array(fix['T'])[:max_len]),
                               'task': fix['task'], 'img_name': fix['name']})
    return processed_fixs


def save_model_train(epoch, args, model, SlowOpt, MidOpt, FastOpt, model_dir, model_name):
    state = {
        "epoch": epoch,
        "args": vars(args),
        "model":
            model.module.state_dict()
            if hasattr(model, "module") else model.state_dict(),
        "optim_slow":
            SlowOpt.state_dict(),
        "optim_mid":
            MidOpt.state_dict(),
        "optim_fast":
            FastOpt.state_dict(),
    }
    torch.save(state, join(model_dir, model_name + '_' + str(epoch) + '.pkg'))


def get_prior_maps(gt_scanpaths, im_w, im_h, visual_angle=24):
    if len(gt_scanpaths) == 0:
        return {}
    task_names = np.unique([traj['task'] for traj in gt_scanpaths])
    all_fixs = []
    prior_maps = {}
    for task in task_names:
        Xs = np.concatenate([
            traj['X'][1:] for traj in gt_scanpaths
            if traj['split'] == 'train' and traj['task'] == task
        ])
        Ys = np.concatenate([
            traj['Y'][1:] for traj in gt_scanpaths
            if traj['split'] == 'train' and traj['task'] == task
        ])
        fixs = np.stack([Xs, Ys]).T.astype(np.int32)
        prior_maps[task] = convert_fixations_to_map(fixs,
                                                    im_w,
                                                    im_h,
                                                    smooth=True,
                                                    visual_angle=visual_angle)
        all_fixs.append(fixs)
    all_fixs = np.concatenate(all_fixs)
    prior_maps['all'] = convert_fixations_to_map(all_fixs,
                                                 im_w,
                                                 im_h,
                                                 smooth=True,
                                                 visual_angle=visual_angle)
    return prior_maps


def convert_fixations_to_map(fixs,
                             width,
                             height,
                             return_distribution=True,
                             smooth=True,
                             visual_angle=16):
    assert len(fixs) > 0, 'Empty fixation list!'

    fmap = np.zeros((height, width))
    for x, y in fixs:
        fmap[y, x] += 1

    if smooth:
        fmap = filters.gaussian_filter(fmap, sigma=visual_angle)

    if return_distribution:
        fmap /= (fmap.sum() + 1e-8)

    return fmap
