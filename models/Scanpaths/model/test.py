import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pickle
import numpy as np
import scipy.stats

import time
import os
import argparse
from os.path import join
from tqdm import tqdm
import datetime
import json
import sys

# from dataset.dataset import AiR, AiR_evaluation
from .dataset.dataset import COCO_Search18, COCO_Search18_evaluation, COCO_Search18_rl
from .models.baseline_attention_multihead import baseline
from .utils.evaluation import human_evaluation, evaluation
from .utils.utils import seed_everything, get_prior_maps
from .utils.metrics import postprocessScanpaths, get_seq_score, get_seq_score_time, get_semantic_seq_score, \
    compute_spatial_metrics_by_step, get_semantic_seq_score_time, compute_mm
from .utils.logger import Logger
from .models.sampling import Sampling


def get_args():
    parser = argparse.ArgumentParser(description="Scanpath prediction for images")
    parser.add_argument("--mode", type=str, default="test", help="Selecting running mode")
    parser.add_argument("--img_dir", type=str, default="datasets/COCO-Search18/images",
                        help="Directory to the image data (stimuli)")
    parser.add_argument("--fix_dir", type=str, default="datasets/COCO-Search18",
                        help="Directory to the raw fixation file")
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
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = get_args()

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load logger
    log_dir = args.evaluation_dir
    # checkpoints_dir = os.path.join(log_dir, "checkpoints")
    checkpoints_dir = args.evaluation_dir
    log_file = os.path.join(log_dir, "log_validation.txt")
    predicts_file = os.path.join(log_dir, "validation_predicts.json")
    logger = Logger(log_file)

    logger.info("The args corresponding to validation process are: ")
    for (key, value) in vars(args).items():
        logger.info("{key:20}: {value:}".format(key=key, value=value))

    validation_dataset = COCO_Search18_evaluation(args.img_dir, args.fix_dir, args.detector_dir,
                                                  type="test", transform=transform,
                                                  detector_threshold=args.detector_threshold)

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=4,
        collate_fn=validation_dataset.collate_func
    )

    object_name = ["bottle", "bowl", "car", "chair", "clock", "cup", "fork", "keyboard", "knife",
                   "laptop", "microwave", "mouse", "oven", "potted plant", "sink", "stop sign",
                   "toilet", "tv"]

    model = baseline(embed_size=512, convLSTM_length=args.max_length, min_length=args.min_length).cuda()

    sampling = Sampling(convLSTM_length=args.max_length, min_length=args.min_length)

    # Load checkpoint to start evaluation.
    # Infer iteration number through file name (it's hacky but very simple), so don't rename
    validation_checkpoint = torch.load(os.path.join(checkpoints_dir, "checkpoint_best.pth"))
    for key in validation_checkpoint:
        if key == "optimizer":
            continue
        else:
            model.load_state_dict(validation_checkpoint[key], strict=False)

    if len(args.gpu_ids) > 1:
        model = nn.DataParallel(model, args.gpu_ids)

    # get the human baseline score
    human_metrics, human_metrics_std, gt_scores_of_each_images = human_evaluation(validation_loader)
    logger.info("The metrics for human performance are: ")
    for metrics_key in human_metrics.keys():
        for (key, value) in human_metrics[metrics_key].items():
            logger.info("{metrics_key:10}-{key:15}: {value:.4f} +- {std:.4f}".format
                        (metrics_key=metrics_key, key=key, value=value, std=human_metrics_std[metrics_key][key]))

    model.eval()
    repeat_num = args.eval_repeat_num
    all_gt_fix_vectors = []
    all_predict_fix_vectors = []
    predict_results = []
    prediction_scanpaths = []
    with tqdm(total=len(validation_loader) * repeat_num) as pbar_val:
        for i_batch, batch in enumerate(validation_loader):
            tmp = [batch["images"], batch["fix_vectors"], batch["attention_maps"], batch["tasks"],
                   batch["img_names"], batch["subjects"]]
            tmp = [_ if not torch.is_tensor(_) else _.cuda() for _ in tmp]
            images, gt_fix_vectors, attention_maps, tasks, img_names, subjects = tmp

            N, C, H, W = images.shape

            if args.ablate_attention_info:
                attention_maps *= 0

            with torch.no_grad():
                predict = model(images, attention_maps, tasks)

            log_normal_mu = predict["log_normal_mu"]
            log_normal_sigma2 = predict["log_normal_sigma2"]
            all_actions_prob = predict["all_actions_prob"]

            for trial in range(repeat_num):
                all_gt_fix_vectors.extend(gt_fix_vectors)

                samples = sampling.random_sample(all_actions_prob, log_normal_mu, log_normal_sigma2)
                prob_sample_actions = samples["selected_actions_probs"]
                durations = samples["durations"]
                sample_actions = samples["selected_actions"]
                sampling_random_predict_fix_vectors, _, _ = sampling.generate_scanpath(
                    images, prob_sample_actions, durations, sample_actions)
                all_predict_fix_vectors.extend(sampling_random_predict_fix_vectors)

                for index in range(N):
                    predict_result = dict()
                    one_sampling_random_predict_fix_vectors = sampling_random_predict_fix_vectors[index]
                    fix_vector_array = np.array(one_sampling_random_predict_fix_vectors.tolist())
                    predict_result["img_names"] = img_names[index]
                    predict_result["task"] = object_name[tasks[index]]
                    predict_result["repeat_id"] = trial + 1
                    predict_result["X"] = list(fix_vector_array[:, 0])
                    predict_result["Y"] = list(fix_vector_array[:, 1])
                    predict_result["T"] = list(fix_vector_array[:, 2] * 1000)
                    predict_result["length"] = len(predict_result["X"])
                    predict_results.append(predict_result)

                    prediction_scanpaths.append({
                        'X': predict_result["X"],
                        'Y': predict_result["Y"],
                        'T': predict_result["T"],
                        'subject': subjects[index],
                        'name': img_names[index],
                        'task': object_name[tasks[index]],
                        'condition': "present"
                    })

                pbar_val.update(1)

    print("[Scanpaths] Running evaluation metrics...")

    cur_metrics, cur_metrics_std, _ = evaluation(all_gt_fix_vectors, all_predict_fix_vectors)

    fixation_path = join(args.fix_dir, 'coco_search18_fixations_TP_test.json')
    with open(fixation_path) as json_file:
        human_scanpaths = json.load(json_file)
    test_target_trajs = list(
        filter(lambda x: x['split'] == 'test' and x['condition'] == 'present', human_scanpaths))
    t_dict = {}
    for traj in test_target_trajs:
        key = 'test-{}-{}-{}-{}'.format(traj['condition'], traj['task'],
                                        traj['name'][:-4], traj['subject'])

        t_dict[key] = np.array(traj['T'])

    fix_clusters = np.load(join(os.path.dirname(args.evaluation_dir), 'data/clusters_test.npy'),
                           allow_pickle=True).item()

    print("Calculating Sequence Score...")
    seq_score = get_seq_score(prediction_scanpaths, fix_clusters, args.max_length)

    print('Sequence Score : {:.3f}\n'.format(seq_score))

    print("Calculating Sequence Score with Duration...")
    seq_score_t = get_seq_score_time(prediction_scanpaths, fix_clusters, args.max_length, t_dict)

    print('Sequence Score with Duration : {:.3f}\n'.format(seq_score_t))

    semantics_root = join(os.path.dirname(args.evaluation_dir), 'data/SemSS')
    sem_file = 'test_TP_Sem.pkl'
    sem_path = join(semantics_root, sem_file)
    segmentation_map_dir = join(semantics_root, 'segmentation_maps')

    with open(sem_path, "rb") as f:
        fixations_dict = pickle.load(f)

    print("Calculating Semantic Sequence Score...")
    sem_seq_score = get_semantic_seq_score(prediction_scanpaths, fixations_dict, args.max_length, segmentation_map_dir)
    print('Semantic Sequence Score: {:.3f}\n'.format(sem_seq_score))
    cur_metrics["SemSS"] = sem_seq_score

    print("Calculating Semantic Sequence Score with Duration...")
    sem_seq_score_t = get_semantic_seq_score_time(prediction_scanpaths, fixations_dict, args.max_length,
                                                  segmentation_map_dir)
    print('Semantic Sequence Score with Duration: {:.3f}\n'.format(sem_seq_score_t))

    print("Calculating IG, CC, NSS, AUC...")
    with open(os.path.join(args.fix_dir, 'coco_search18_fixations_TP_train.json')) as f:
        human_scanpaths_train = json.load(f)
    with open(os.path.join(args.fix_dir, 'coco_search18_fixations_TP_validation.json')) as f:
        human_scanpaths_valid = json.load(f)
    hs = human_scanpaths_train + human_scanpaths_valid + human_scanpaths
    hs = list(filter(lambda x: x['fixOnTarget'] or x['condition'] == 'absent', hs))
    hs = list(filter(lambda x: x['condition'] == 'present', hs))

    prior_maps = get_prior_maps(hs, im_w=512, im_h=320)
    keys = list(prior_maps.keys())
    device = torch.device(f'cuda:{args.gpu_ids[0]}')
    for k in keys:
        prior_maps[k] = torch.tensor(prior_maps.pop(k)).to(device)

    ig, cc, nss, auc = compute_spatial_metrics_by_step(prediction_scanpaths, test_target_trajs, 512, 320, prior_maps)
    print("IG: {:.3f}".format(ig))
    print("CC: {:.3f}".format(cc))
    print("NSS: {:.3f}".format(nss))
    print("AUC: {:.3f}".format(auc))

    with open(predicts_file, 'w') as f:
        json.dump(predict_results, f, indent=2)

    for metrics_key, metrics_dict in cur_metrics.items():
        if isinstance(metrics_dict, dict):
            for metric_name, metric_value in metrics_dict.items():
                print(f"{metrics_key} {metric_name.capitalize()}: {metric_value:.3f}")
        else:
            print(f"{metrics_key}: {metrics_dict:.3f}")
