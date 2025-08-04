import argparse
import os
import random
from os.path import join
import numpy as np
import torch
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
    num = 0
    for fix in fixations:
        if len(fix['X']) <= max_len:
            processed_fixs.append({'tgt_seq_y': torch.tensor(np.array(fix['Y'])[:max_len]),
                                   'tgt_seq_x': torch.tensor(np.array(fix['X'])[:max_len]),
                                   'tgt_seq_t': torch.tensor(np.array(fix['T'])[:max_len]),
                                   'task': fix['task'], 'img_name': fix['name']})
        else:
            num += 1
            processed_fixs.append({'tgt_seq_y': torch.tensor(np.array(fix['Y'])[-max_len:]),
                                   'tgt_seq_x': torch.tensor(np.array(fix['X'])[-max_len:]),
                                   'tgt_seq_t': torch.tensor(np.array(fix['T'])[-max_len:]),
                                   'task': fix['task'], 'img_name': fix['name']})
    print("Has:%d scanpath over length" % num)
    return processed_fixs


def save_model_train(epoch, args, model, SlowOpt, MidOpt, FastOpt, model_dir, model_name, save_path="",
                     only_trainable=True):
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
    if only_trainable:
        weight_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        name = model.named_parameters()
        state = {
            "epoch": epoch,
            "args": vars(args),
            "model": {n: weight_dict[n] for n, p in name if p.requires_grad},
            "optim_slow":
                SlowOpt.state_dict(),
            "optim_mid":
                MidOpt.state_dict(),
            "optim_fast":
                FastOpt.state_dict(),
        }
    if save_path:
        torch.save(state, save_path)
    else:
        torch.save(state, join(model_dir, model_name + '_' + str(epoch) + '.pkg'))


def cutFixOnTarget(trajs, target_annos):
    processed_trajs = []
    task_names = np.unique([traj['task'] for traj in trajs])
    if 'condition' in trajs[0].keys():
        trajs = list(filter(lambda x: x['condition'] == 'present', trajs))
    if len(trajs) == 0:
        return
    for task in task_names:
        task_trajs = list(filter(lambda x: x['task'] == task, trajs))
        num_steps_task = np.ones(len(task_trajs), dtype=np.uint8)
        for i, traj in enumerate(task_trajs):
            key = traj['task'] + '_' + traj['img_name']
            bbox = target_annos[key]
            traj_len = get_num_step2target(traj['tgt_seq_x'], traj['tgt_seq_y'], bbox)
            num_steps_task[i] = traj_len
            traj['tgt_seq_x'] = traj['tgt_seq_x'][:traj_len]
            traj['tgt_seq_y'] = traj['tgt_seq_y'][:traj_len]
            traj['tgt_seq_t'] = traj['tgt_seq_t'][:traj_len]
            if traj_len != 100:
                processed_trajs.append(traj)
    print('data cuted')
    return processed_trajs


def get_num_step2target(X, Y, bbox):
    X, Y = np.array(X), np.array(Y)
    on_target_X = np.logical_and(X > bbox[0], X < bbox[0] + bbox[2])
    on_target_Y = np.logical_and(Y > bbox[1], Y < bbox[1] + bbox[3])
    on_target = np.logical_and(on_target_X, on_target_Y)
    if np.sum(on_target) > 0:
        first_on_target_idx = np.argmax(on_target)
        return first_on_target_idx + 1
    else:
        return 100


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