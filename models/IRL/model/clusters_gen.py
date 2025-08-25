import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.cluster import MeanShift, estimate_bandwidth

annotation_root = "datasets/COCO-Search18"

with open(os.path.join(annotation_root, 'coco_search18_fixations_TP_train.json')) as f:
    human_scanpaths_train = json.load(f)
with open(os.path.join(annotation_root, 'coco_search18_fixations_TP_validation.json')) as f:
    human_scanpaths_valid = json.load(f)
with open(os.path.join(annotation_root, 'coco_search18_fixations_TP_test.json')) as f:
    human_gt = json.load(f)

scanpaths = human_scanpaths_train + human_scanpaths_valid + human_gt

xs, ys = [], []
valid_scanpaths = []

for i, scanpath in enumerate(tqdm(scanpaths, desc="Filtering scanpaths")):
    try:
        x = np.array(scanpath["X"], dtype=np.float32).reshape(-1, 1)
        y = np.array(scanpath["Y"], dtype=np.float32).reshape(-1, 1)

        if x.shape[0] == 0 or x.shape != y.shape:
            continue

        xs.append(x)
        ys.append(y)
        valid_scanpaths.append(scanpath)

    except Exception as e:
        print(f"[Skip] Error parsing scanpath {i}: {e}")
        continue

gt_gaze = np.hstack((np.vstack(xs), np.vstack(ys)))
print("[Log] gt_gaze shape:", gt_gaze.shape)

bandwidth = estimate_bandwidth(gt_gaze)
# bandwidth = 97.6524
factors = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
rates = []

def scanpath2clusters(meanshift, scanpath):
    string = []
    for x, y in zip(scanpath['X'], scanpath['Y']):
        symbol = meanshift.predict([[x, y]])[0]
        string.append(symbol)
    return string

def improved_rate(meanshift, scanpaths):
    Nc = len(meanshift.cluster_centers_)
    Nb, Nw = 0, 0
    for scanpath in scanpaths:
        string = scanpath2clusters(meanshift, scanpath)
        for i in range(len(string) - 1):
            if string[i] == string[i + 1]:
                Nw += 1
            else:
                Nb += 1
    return (Nb - Nw) / Nc

for factor in factors:
    bd = bandwidth * factor
    print(f"[Clustering] Trying bandwidth × {factor:.2f} → {bd:.4f}", flush=True)
    ms = MeanShift(bandwidth=bd)
    ms.fit(gt_gaze)
    rate = improved_rate(ms, valid_scanpaths)
    rates.append(rate)

best_bd = factors[np.argmax(rates)] * bandwidth
print(f"[Result] Best bandwidth = {best_bd:.4f}")
best_ms = MeanShift(bandwidth=best_bd)
best_ms.fit(gt_gaze)

os.makedirs("models/IRL/data", exist_ok=True)
np.save("models/IRL/data/clusters_test.npy", best_ms.cluster_centers_)
print(f"[Saved] clusters_test.npy → {best_ms.cluster_centers_.shape}")

gt_strings = []
for scanpath in tqdm(human_gt, desc="Generating gt_strings for test set"):
    gt_string = scanpath2clusters(best_ms, scanpath)
    gt_strings.append(gt_string)

with open("models/IRL/data/gt_strings_test.json", "w") as f:
    json.dump(gt_strings, f)
print("[Saved] gt_strings_test.json")
