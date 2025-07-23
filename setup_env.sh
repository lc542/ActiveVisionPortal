#!/bin/bash

module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load cudnn/8.9.5.29
module load python/3.10
module load opencv/4.11.0

ENV_PATH=~/envs/activevision
if [ ! -d "$ENV_PATH" ]; then
    echo ">>> Creating virtualenv at $ENV_PATH"
    python -m venv $ENV_PATH
fi

source $ENV_PATH/bin/activate

python -m pip install --upgrade pip setuptools wheel

python -m pip install git+https://github.com/openai/CLIP.git --no-build-isolation

python -m pip install torch==2.6.0 torchvision==0.21.0
python -m pip install git+https://github.com/facebookresearch/detectron2.git --no-build-isolation

python -m pip install absl-py==2.3.0 addict==2.4.0 antlr4-python3-runtime==4.9.3 astunparse==1.6.3 cloudpickle==3.1.1 cython==3.1.1 docopt==0.6.2 filelock==3.13.1 fvcore==0.1.5.post20221221 iopath==0.1.9 mmcv==1.1.4 mmengine==0.10.7 multimatch-gaze==0.1.2 opencv-python==4.11.0.86 pycocotools==2.0.9 pillow==11.0.0 scikit-image==0.25.2 scipy==1.15.1 timm==1.0.15

python -m pip install transformers==4.52.4 tokenizers==0.21.1 huggingface-hub==0.33.0 sentence-transformers==4.1.0

python -m pip install black==25.1.0 click==8.2.1 fonttools==4.58.1 jinja2==3.1.4 lazy-loader==0.4 markdown==3.8 matplotlib==3.10.3 numpy==2.1.1 packaging==25.0 pandas==2.2.3 regex==2024.11.6 requests==2.32.3 rich==14.0.0 seaborn==0.13.2 tabulate==0.9.0 tqdm==4.67.1 wget==3.2 yacs==0.1.8 yapf==0.43.0

python -m pip install tensorflow==2.17.0 tensorboard==2.17.0 tensorboard-data-server==0.7.2 tensorflow-io-gcs-filesystem==0.32.0 wrapt==1.17.2 opt-einsum==3.4.0 ml-dtypes==0.4.0 flatbuffers==25.2.10 termcolor==3.1.0 gast==0.6.0 google-pasta==0.2.0 protobuf==4.25.8 keras==3.10.0

python -m pip install git-filter-repo==2.47.0 libclang==18.1.1 llvmlite==0.44.0 ninja==1.11.1.4 numba==0.61.2 optree==0.13.0