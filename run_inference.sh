#!/usr/bin/env bash
set -euo pipefail

# setting up the environment
if [ ! -d .venv ]; then
  uv venv --python 3.11
fi
source .venv/bin/activate
uv pip install -e .[cuda]
uv pip install vllm

export HF_TOKEN=$MILABENCH_HF_TOKEN

export MILABENCH_BASE="$PWD"
export MILABENCH_CONFIG="$PWD/config/standard.yaml"
export MILABENCH_SYSTEM="$PWD/config/system.yaml"
export MILABENCH_SSH=~/.ssh/id_ed25519
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# auto-generate a minimal single-node system.yaml if this machine doesn't
# have one yet (GPU capacity/count are auto-detected by milabench itself;
# only the node's name/ip/user are mandatory and can't be auto-discovered
# for a real multi-node cluster, but this is enough for num_machines: 1)
if [ ! -f config/system.yaml ]; then
  cat > config/system.yaml <<EOF
system:
  arch: cuda
  nodes:
    - name: local
      ip: 127.0.0.1
      user: $(whoami)
      main: true
EOF
fi

# clear out any previous run data, so re-runs start clean instead of
# appending to old .data files (milabench opens them in append mode).
# Safe here because this script's results are extracted into their own
# results_inference.csv below before anything else touches runs/ again.
rm -rf "$MILABENCH_BASE/runs"



# installing milabench tests
milabench install --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp,llm-full-mp-nodes-tp,llm-full-mp-nodes-cp,vllm-dense-physics-gpus,vllm-moe-code-gpus

# preparing milabench tests
milabench prepare --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp,vllm-dense-physics-gpus,vllm-moe-code-gpus


# inference tests
milabench run --config config/all.yaml --system config/system.yaml --select vllm-dense-physics-gpus
python compare_to_csv.py --out results_inference.csv

milabench run --config config/all.yaml --system config/system.yaml --select vllm-moe-code-gpus
python compare_to_csv.py --out results_inference.csv

# collecting results
# (must run before vllm_main.py below: compare_to_csv.py overwrites
# results_inference.csv from scratch from runs/, which would wipe out
# vllm_main.py's appended row)
python compare_to_csv.py --out results_inference.csv

# vllm_main.py (standalone custom load test, not a milabench-managed benchmark)
# needs its own vLLM server, since it's just an HTTP client
# same on-disk weights used for the FSDP/TP/CP fine-tuning experiments
# (checkpointer.checkpoint_dir={milabench_data}/llama3_70B there too)
VLLM_MAIN_GPU_COUNT=$(nvidia-smi -L | wc -l)
vllm serve "$MILABENCH_BASE/data/llama3_70B" --served-model-name model --port 8000 --tensor-parallel-size "$VLLM_MAIN_GPU_COUNT" &
VLLM_MAIN_SERVER_PID=$!

# a 70B model takes a while to load, give it up to 30 minutes to come up
for i in $(seq 1 180); do
  if curl -s -o /dev/null "http://127.0.0.1:8000/health"; then
    break
  fi
  sleep 10
done

python vllm_main.py --host 127.0.0.1 --port 8000 --model model -n 10000 --max-tokens 1000 --csv results_inference.csv --run-name vllm-main-n10000-max1000 --bench vllm-main-n10000-max1000

python vllm_main.py --host 127.0.0.1 --port 8000 --model model -n 100 --max-tokens 10000 --csv results_inference.csv --run-name vllm-main-n100-max10000 --bench vllm-main-n100-max10000

kill "$VLLM_MAIN_SERVER_PID"
wait "$VLLM_MAIN_SERVER_PID" 2>/dev/null
