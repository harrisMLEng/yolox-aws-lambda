#!/usr/bin/env bash


VENV_PATH=$(poetry env info --path)
if [[ ! -d $VENV_PATH ]]; then
  poetry config virtualenvs.in-project true --local
fi
poetry install
poetry shell