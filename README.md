# ActiveVisionPortal

## About

This project develops an open portal for goal-directed vision research.

## Datasets

* **COCO-Search18** dataset is available at [https://sites.google.com/view/cocosearch/home](https://sites.google.com/view/cocosearch/home).

## Environment Setup

The environment dependencies are specified in [`environment.yaml`](./environment.yaml). To set up the environment using `conda`, run:

```bash
  conda env create -f environment.yaml
  conda activate ActiveVision
```

Alternatively, for Compute Canada users, see the [detailed section](#setting-up-on-compute-canada) below.

## Running the Framework

The main entry point is [`main.py`](./main.py), which provides a unified interface for training and evaluating different gaze prediction models.

### Example Commands

* **List all available models:**

```bash
  python main.py --list_models
```

* **Train a model:**

```bash
  python main.py --model <model_name> --train --dataset datasets/COCO-Search18
```

* **Evaluate a model:**

```bash
  python main.py --model <model_name> --eval --dataset datasets/COCO-Search18
 ```

* **Show model-specific help:**

```bash
  python main.py --model <model_name> --help_model
```

You may also specify the dataset directory via `--dataset`, e.g., `--dataset datasets/COCO-Search18`.

## Required Files

Due to storage constraints, some required files are hosted on Compute Canada. **Please place them in exactly the same directory structure as referenced in the code.** For example:

```
ActiveVisionPortal/
├── datasets/
│   └── COCO-Search18/
│   └── ...
├── models/
│   └── model 1/
│       └── checkpoint(s)/
│       └── data/
│       └── pretrained_models(if applicable)/
│       └── model/
│       └── entry.py
│       └── ...
│   └── model 2/
│   └── ...
```

## Setting Up on Compute Canada

### Step 1: Prepare Project Directory
Please place the entire project directory in a location of your choice on Compute Canada, which matches the directory structure shown in [Required Files](#required-files).

For example, you can use: `~/scratch/ActiveVisionPortal/`

### Step 2: Create Environment

* Run the provided setup script:   

```bash
  bash setup_env.sh
```

* Compile Custom CUDA Operators: 
```bash
  cd ~/scratch/ActiveVisionPortal/models/HAT/model/pixel_decoder/ops/
  dos2unix make.sh
```

* Allocate a GPU node:
```bash
  salloc --gres=gpu:1 --mem=16G --cpus-per-task=4 --time=01:00:00
```

* Inside the GPU shell:
```bash
  sh make.sh
  exit
```

### Step 3: Running
Navigate to your project root (e.g., where `main.py` is located):

```bash
  cd ~/scratch/ActiveVisionPortal
```
Then follow the instructions in the [Running the Framework section](#running-the-framework) below to train or evaluate a model.