# When Languages Disagree: Self-Evolving Multilingual LLM Judges

**step1**: Prepare data and yaml file (e.g., xcopa) under unified_data folder
```
{
  "lang": {
    "premise": "...",
    "choice1": "...",
    "choice2": "...",
    "question": "effect",
    "label": 1,
    "idx": 0,
    "changed": false,
    "output": "...",
    "output_answer": true
  }
}
```

**step2**: Launch the model
```
python -m sglang.launch_server \
  --model-path PATH \
  --tp-size 1 \
  --trust-remote-code \
  --port 30000 \
  > sglang.log 2>&1 &
```

**step3**: run script
```
sh run_judge_evolve_multi_round.sh
```

## Citations
Please cite our paper if you are using this dataset.
```

```