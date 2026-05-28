#!/usr/bin/env bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
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
#0.01和0.1被kill了
EPS_LIST=(0.01)
LR_LIST=(0.1)
Q_LIST=(20)
# TIMES=(1,2,3)
STEPS=(40)
METHOD="adam"
MAX_JOBS=2 
job_count=0
# ====== 网格搜索 ======

for eps in "${EPS_LIST[@]}"; do
  for lr in "${LR_LIST[@]}"; do
    for q in "${Q_LIST[@]}"; do
    for s in "${STEPS[@]}"; do

          name="everylora_${s}_eps${eps}_lr${lr}_q${q}"

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

# echo "===== BEST RESULT ====="
# grep "task accuracy" output/*.log | sort -t ':' -k2 -nr | head -n 5