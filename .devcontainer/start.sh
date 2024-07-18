#!/bin/bash

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "Conda initialization script not found. Please check your conda installation."
    exit 1
fi

if ! conda env list | grep -q 'yolox-aws-lambda'; then
    conda env create --file environment.yml
fi

conda env list