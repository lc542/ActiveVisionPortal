import numpy as np
import os
import random
import torch
import argparse
from os.path import join


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
