#!/bin/bash
# Part (d): 3 models x 3 learning rates, 100 epochs each, run one after another.
cd "$(dirname "$0")"
PY=../../.venv/bin/python
for v in resnet18 no_residual no_batchnorm; do
  for lr in 0.003 0.01 0.03; do
    if [ -f runs/${v}_lr${lr}/model-100.ckpt ]; then continue; fi
    echo "===== $v lr=$lr  $(date)"
    $PY -u main.py --variant $v --lr $lr 2>&1 \
      | perl -pe 'BEGIN { $| = 1 } s/\r/\n/g' | grep --line-buffered -v '^Batch'
  done
done
echo "===== ALL DONE $(date)"
