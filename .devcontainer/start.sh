#!/usr/bin/env bash


VENV_PATH=$(poetry env info --path)
if [[ ! -d $VENV_PATH ]]; then
  poetry config virtualenvs.in-project true --local
fi


if [ ! -d "YOLOX" ]; then
    git clone --depth=1 https://github.com/Megvii-BaseDetection/YOLOX.git
fi



poetry install
poetry shell

cd YOLOX
poetry run pip3 install .

# python3 setup.py

# cd YOLOX
# if ! poetry run pip3 install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2; then
#     exit 1
# fi
# if ! poetry run pip3 install .; then
#     cd ..
#     poetry shell
#     exit 1
# fi
# rm -rf YOLOX