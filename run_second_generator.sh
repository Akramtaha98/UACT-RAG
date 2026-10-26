#!/usr/bin/env bash
# Robustness check (post hoc): rerun the same 310 test questions with a second, larger generator. The estimator, controller and
# baseline thresholds stay FROZEN (fitted with Qwen3-8B), so this tests whether the conclusions depend on the generator.
#   G=1 nohup bash run_second_generator.sh > logs/run_second_gen.out 2>&1 &     (use the vLLM environment; needs ~70 GB free on the GPU)
set -euo pipefail
cd "$(dirname "$0")"
G=${G:-1}; GEN=${GEN:-Qwen/Qwen3-32B}; DATA=data/techqa
CUDA_VISIBLE_DEVICES=$G python scripts/run_eval.py --data $DATA --split test --systems B2 B6 B9 B8 UACT \
  --set models.generator=$GEN models.vllm_gpu_mem_util=0.75 eval.use_llm_judge=false --tag _gen32b > logs/test_gen32b.log 2>&1
grep -E "^(B[0-9]|UACT)" logs/test_gen32b.log || true
echo "SECOND GENERATOR DONE $(date)"
