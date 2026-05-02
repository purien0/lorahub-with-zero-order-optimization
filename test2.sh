#!/usr/bin/env bash

set -euo pipefail

mkdir -p output

run_experiment() {
  local name="$1"
  shift

  echo "===== Running ${name} ====="
  python example.py "$@" | tee "output/${name}.log"
}

# run_experiment baseline --method baseline --steps 40
# run_experiment momentum_default --method momentum --steps 40 --eps 0.05 --lr 0.1 --q 5 --beta 0.9
# run_experiment adam_default --method adam --steps 40 --eps 0.05 --lr 0.1 --q 5 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8

EPS_LIST=(0.05)
LR_LIST=( 0.2)
Q_LIST=(10)

STEPS=(40 100)
METHOD="adam"

# ====== 网格搜索 ======

for eps in "${EPS_LIST[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    for q in "${Q_LIST[@]}"; do
        for s in "${STEPS[@]}"; do

          name="adamclip_step${s}_eps${eps}_lr${lr}_q${q}"

          run_experiment "$name" \
            --method ${METHOD} \
            --steps ${s} \
            --eps ${eps} \
            --lr ${lr} \
            --q ${q} \
            --beta1 0.9 \
            --beta2 0.999 \
            --adam_eps 1e-8
        done
    done
  done
done

echo "===== BEST RESULT ====="