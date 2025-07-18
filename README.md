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
project_root/
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