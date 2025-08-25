import argparse
from os.path import join
import json
import numpy as np
import torch
import os

from TEMP.saliency_metrics import cc
from .models import Transformer
from .CLIPGaze import CLIPGaze
from .utils import seed_everything, get_prior_maps
from .metrics import postprocessScanpaths, get_seq_score, get_seq_score_time, get_ed, get_semantic_ed, \
    get_semantic_seq_score, get_semantic_seq_score_time, compute_mm, compute_spatial_metrics_by_step
from tqdm import tqdm
import warnings
import pickle

warnings.filterwarnings("ignore")


def run_model(model, src, task, device="cuda:0", im_h=20, im_w=32, project_num=16, num_samples=1):
    task = torch.tensor(task.astype(np.float32)).to(device).unsqueeze(0).repeat(num_samples, 1)
    firstfix = torch.tensor([(im_h // 2) * project_num, (im_w // 2) * project_num]).unsqueeze(0).repeat(num_samples, 1)
    with torch.no_grad():
        token_prob, ys, xs, ts = model(src=src, tgt=firstfix, task=task)
    token_prob = token_prob.detach().cpu().numpy()
    ys = ys.cpu().detach().numpy()
    xs = xs.cpu().detach().numpy()
    ts = ts.cpu().detach().numpy()
    scanpaths = []
    for i in range(num_samples):
        ys_i = [(im_h // 2) * project_num] + list(ys[:, i, 0])[1:]
        xs_i = [(im_w // 2) * project_num] + list(xs[:, i, 0])[1:]
        ts_i = list(ts[:, i, 0])
        token_type = [0] + list(np.argmax(token_prob[:, i, :], axis=-1))[1:]
        scanpath = []
        for tok, y, x, t in zip(token_type, ys_i, xs_i, ts_i):
            if tok == 0:
                scanpath.append([min(im_h * project_num - 2, y), min(im_w * project_num - 2, x), t])
            else:
                break
        scanpaths.append(np.array(scanpath))
    return scanpaths


def test(args):
    trained_model = args.trained_model
    device = torch.device('cuda:{}'.format(args.cuda))
    transformer = Transformer(nhead=args.nhead, d_model=args.hidden_dim,
                              num_decoder_layers=args.num_decoder, dim_feedforward=args.hidden_dim,
                              device=device).to(device)

    model = CLIPGaze(transformer, spatial_dim=(args.im_h, args.im_w), max_len=args.max_len, device=device).to(device)
    model.load_state_dict(torch.load(trained_model, map_location=device))
    model.eval()

    dataset_root = args.dataset_dir
    clipgaze_data_dir = 'models/CLIPGaze/data'
    img_ftrs_dir = join(clipgaze_data_dir, 'vit-L-14/featuremaps')
    if args.condition == 'absent':
        img_ftrs_dir = args.img_ftrs_dir_absent
    max_len = args.max_len
    fixation_path = join(dataset_root, 'coco_search18_fixations_TP_test.json')
    if args.condition == 'absent':
        fixation_path = join(dataset_root, 'coco_search18_fixations_TA_test.json')
    with open(fixation_path) as json_file:
        human_scanpaths = json.load(json_file)
    test_target_trajs = list(
        filter(lambda x: x['split'] == 'test' and x['condition'] == args.condition, human_scanpaths))
    if args.zerogaze:
        test_target_trajs = list(filter(lambda x: x['task'] == args.task.replace('_', ' '), test_target_trajs))
        print("Zero Gaze on", args.task.replace('_', ' '))
    t_dict = {}
    for traj in test_target_trajs:
        key = 'test-{}-{}-{}-{}'.format(traj['condition'], traj['task'],
                                        traj['name'][:-4], traj['subject'])
        t_dict[key] = np.array(traj['T'])

    test_task_img_pairs = np.unique(
        [traj['task'] + '_' + traj['name'] + '_' + traj['condition'] for traj in test_target_trajs])
    embedding_dict = np.load(open(args.embedding_dir, mode='rb'), allow_pickle=True).item()
    pred_list = []
    print('Generating {} scanpaths per test case...'.format(args.num_samples))

    for target_traj in tqdm(test_task_img_pairs):
        task_name, name, condition = target_traj.split('_')
        image_ftrs = [torch.load(join(img_ftrs_dir, task_name.replace(' ', '_'), name.replace('jpg', 'pth')))]
        task_emb = embedding_dict[task_name]
        scanpaths = run_model(model=model, src=image_ftrs, task=task_emb, device=device, num_samples=args.num_samples)
        for idx, scanpath in enumerate(scanpaths):
            pred_list.append((task_name, name, condition, idx + 1, scanpath))

    print("[CLIPGaze] Running evaluation metrics...")

    predictions = postprocessScanpaths(pred_list)
    fix_clusters = np.load(join(clipgaze_data_dir, 'clusters_test.npy'), allow_pickle=True).item()
    print("Calculating Sequence Score...")
    seq_score = get_seq_score(predictions, fix_clusters, max_len)
    print('Sequence Score : {:.3f}\n'.format(seq_score))

    print("Calculating Sequence Score with Duration...")
    seq_score_t = get_seq_score_time(predictions, fix_clusters, max_len, t_dict)
    print('Sequence Score with Duration : {:.3f}\n'.format(seq_score_t))

    print("Calculating FED...")
    FED = get_ed(predictions, fix_clusters, max_len)
    print('FED: {:.3f}\n'.format(FED))

    if args.condition == 'present':
        with open('models/Gazeformer/data/SemSS/test_TP_Sem.pkl', "rb") as r:
            fixations_dict = pickle.load(r)
            r.close()
    elif args.condition == 'absent':
        with open('models/Gazeformer/data/SemSS/test_TA_Sem.pkl', "rb") as r:
            fixations_dict = pickle.load(r)
            r.close()

    print("Calculating Semantic Sequence Score...")
    SemSS = get_semantic_seq_score(predictions, fixations_dict, max_len,
                                   'models/Gazeformer/data/SemSS/segmentation_maps')
    print('Semantic Sequence Score: {:.3f}\n'.format(SemSS))

    print("Calculating Semantic Sequence Score with Duration...")
    SemSS_t = get_semantic_seq_score_time(predictions, fixations_dict, max_len, 'models/Gazeformer/data/SemSS/segmentation_maps')
    print('Semantic Sequence Score with Duration: {:.3f}\n'.format(SemSS_t))

    print("Calculating SemFED...")
    SemFED = get_semantic_ed(predictions, fixations_dict, max_len, 'models/Gazeformer/data/SemSS/segmentation_maps')
    print('SemFED: {:.3f}\n'.format(SemFED))

    print("\nCalculating MultiMatch...")
    if args.condition == 'absent':
        for x in test_target_trajs:
            x['X'] = [a / 1680 * 512 for a in x['X']]
            x['Y'] = [a / 1050 * 320 for a in x['Y']]

    mm = compute_mm(test_target_trajs, predictions, 512, 320)
    vec, dir_, len_, pos = mm[:4]
    mm_mean = mm[:-1].mean()

    print("MultiMatch:")
    print(f"  Vector:    {vec:.4f}")
    print(f"  Direction: {dir_:.4f}")
    print(f"  Length:    {len_:.4f}")
    print(f"  Position:  {pos:.4f}")
    print(f"  MultiMatch Mean:  {mm_mean:.4f}")

    print("Calculating IG, CC, NSS, AUC...")

    with open(os.path.join(dataset_root, 'coco_search18_fixations_TP_train.json')) as f:
        human_scanpaths_train = json.load(f)
    with open(os.path.join(dataset_root, 'coco_search18_fixations_TP_validation.json')) as f:
        human_scanpaths_valid = json.load(f)
    with open(os.path.join(dataset_root, 'coco_search18_fixations_TP_test.json')) as f:
        human_gt = json.load(f)
    hs = human_scanpaths_train + human_scanpaths_valid + human_gt
    hs = list(filter(lambda x: x['fixOnTarget'] or x['condition'] == 'absent', hs))
    hs = list(filter(lambda x: x['condition'] == args.condition, hs))

    prior_maps = get_prior_maps(hs, im_w=512, im_h=320)
    keys = list(prior_maps.keys())
    for k in keys:
        prior_maps[k] = torch.tensor(prior_maps.pop(k)).to(device)

    ig, cc, nss, auc = compute_spatial_metrics_by_step(predictions, test_target_trajs, 512, 320, prior_maps)
    print("IG: {:.3f}".format(ig))
    print("CC: {:.3f}".format(cc))
    print("NSS: {:.3f}".format(nss))
    print("AUC: {:.3f}".format(auc))

    return seq_score, seq_score_t, FED, SemSS, SemSS_t, SemFED, mm_mean, ig, cc, nss, auc


def main(args):
    seed_everything(args.seed)
    seq_score, seq_score_t, FED, SemSS, SemSS_t, SemFED, mm_mean, ig, cc, nss, auc = test(args)
    # print('Sequence Score : {:.3f}, FED : {:.3f}'.format(seq_score, FED))
    print(seq_score, seq_score_t, FED, SemSS, SemSS_t, SemFED, mm_mean, ig, cc, nss, auc)
