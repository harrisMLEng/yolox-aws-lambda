#!/usr/bin/env bash

if ! conda env list | grep -q 'yolox-aws-lambda'; then
    conda env create --file environment.yml
fi

source activate base
conda activate yolox-aws-lambda

