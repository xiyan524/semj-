#!/bin/bash

CONFIG=dataset_languages.json
MAX_ROUNDS=$1
CONSISTENCY_THRESHOLD=$2
TEMPERATURES=$3
MODEL=$4
EXTRA_LANG=$5

for file in $(python -c "import json; d=json.load(open('$CONFIG'))['unified_data']; print(' '.join(d.keys()))")
do
  for lang in $(python -c "import json; d=json.load(open('$CONFIG'))['unified_data']['$file']; print(' '.join(d))")
  do
    echo "Running (multi-round) $file | $lang | temp=0"

    python judge_evolve_multi_round_ablation.py \
      --input unified_data/$file \
      --lang $lang \
      --judge-model $MODEL \
      --workers 4\
      --round-parallel 4\
      --extra-lang-count $EXTRA_LANG\
      --max-rounds $MAX_ROUNDS \
      --consistency-threshold $CONSISTENCY_THRESHOLD \
      --out-dir judge_outputs/ \
      --temperature $TEMPERATURES
  done
done
