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
# results_fsdp.csv below before anything else touches runs/ again.
rm -rf "$MILABENCH_BASE/runs"



# installing milabench tests
milabench install --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp,llm-full-mp-nodes-tp,llm-full-mp-nodes-cp,vllm-dense-physics-gpus,vllm-moe-code-gpus

# preparing milabench tests
milabench prepare --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp,vllm-dense-physics-gpus,vllm-moe-code-gpus


# FSDP

# unpacked
MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=2 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs2-unpacked
python compare_to_csv.py --out results_fsdp.csv

MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=4 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs4-unpacked
python compare_to_csv.py --out results_fsdp.csv

MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=8 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs8-unpacked
python compare_to_csv.py --out results_fsdp.csv

MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=16 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs16-unpacked
python compare_to_csv.py --out results_fsdp.csv

MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=32 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs32-unpacked
python compare_to_csv.py --out results_fsdp.csv

MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=64 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs64-unpacked
python compare_to_csv.py --out results_fsdp.csv

# packed
MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=2 MILABENCH_LLM_PACKED=True milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs2-packed
python compare_to_csv.py --out results_fsdp.csv

MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=4 MILABENCH_LLM_PACKED=True milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs4-packed
python compare_to_csv.py --out results_fsdp.csv
