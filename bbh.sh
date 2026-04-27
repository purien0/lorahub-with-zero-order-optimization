#!/usr/bin/env bash

mkdir -p output
python reproduce_bbh.py --method adam --steps 40 --eps 0.05 --lr 0.1 --q 5 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 | tee "output/bbh.log"

