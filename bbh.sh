#!/usr/bin/env bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p output
python reproduce_bbh.py --method adam --steps 40 --eps 0.01 --lr 0.1 --q 20 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 | tee "output/bbh-zolearning3.log"

