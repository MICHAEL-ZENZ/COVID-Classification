#!/usr/bin/env bash
nvidia-smi
conda env create -f environment.yml
conda activate zb
pip install Pillow==6.2.2
python main.py
