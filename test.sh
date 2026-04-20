#!/usr/bin/env bash

set -euo pipefail

mkdir -p output

run_experiment() {
  local name="$1"
  shift

  echo "===== Running ${name} ====="
  python example.py "$@" | tee "output/${name}.log"
}

#run_experiment baseline --method baseline --steps 40
run_experiment momentum_default --method momentum --steps 40 --eps 0.05 --lr 0.1 --q 5 --beta 0.9
run_experiment adam_default --method adam --steps 40 --eps 0.05 --lr 0.1 --q 5 --beta1 0.9 --beta2 0.999 --adam_eps 1e-8
