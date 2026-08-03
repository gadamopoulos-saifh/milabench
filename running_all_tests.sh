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
# appending to old .data files (milabench opens them in append mode)
rm -rf "$MILABENCH_BASE/runs"

# every command's combined stdout+stderr is teed to logs/<name>.txt, so a
# crashed/OOM'd experiment can be inspected afterwards while still showing
# up live on the terminal. Existing logs are overwritten per-experiment
# rather than the whole folder being wiped, so unrelated old logs survive.
LOG_DIR="$MILABENCH_BASE/logs"
mkdir -p "$LOG_DIR"

# without this, the exit status of "cmd | tee" would be tee's (always 0)
set -o pipefail

# log_run <log-name> <command...>
# Env vars have to go through `env` rather than as a "VAR=x log_run ..."
# prefix: bash keeps assignments made in front of a *function* call set in
# the shell afterwards, which would leak the sizer/packed settings into
# every later experiment.
log_run() {
  local name="$1"; shift
  echo "=== $name: $* ==="
  "$@" 2>&1 | tee "$LOG_DIR/$name.txt"
}

# compare_to_csv.py runs after every experiment, so keep all of its (short)
# output in one appended file instead of ~25 near-empty ones
: > "$LOG_DIR/compare_to_csv.txt"
collect_results() {
  python compare_to_csv.py --out results.csv 2>&1 | tee -a "$LOG_DIR/compare_to_csv.txt"
}


# installing milabench tests
log_run install milabench install --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp,llm-full-mp-nodes-tp,llm-full-mp-nodes-cp,vllm-dense-physics-gpus,vllm-moe-code-gpus

# preparing milabench tests
log_run prepare milabench prepare --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp,vllm-dense-physics-gpus,vllm-moe-code-gpus


# FSDP

# unpacked
log_run fsdp-bs2-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=2 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs2-unpacked
collect_results

log_run fsdp-bs4-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=4 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs4-unpacked
collect_results

log_run fsdp-bs8-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=8 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs8-unpacked
collect_results

log_run fsdp-bs16-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=16 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs16-unpacked
collect_results

log_run fsdp-bs32-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=32 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs32-unpacked
collect_results

log_run fsdp-bs64-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=64 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs64-unpacked
collect_results

# packed
log_run fsdp-bs2-packed env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=2 MILABENCH_LLM_PACKED=True milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs2-packed
collect_results

log_run fsdp-bs4-packed env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=4 MILABENCH_LLM_PACKED=True milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-fsdp --run-name fsdp-bs4-packed
collect_results


# TP

# unpacked
log_run tp-bs2-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=2 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-tp --run-name tp-bs2-unpacked
collect_results

log_run tp-bs4-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=4 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-tp --run-name tp-bs4-unpacked
collect_results

log_run tp-bs8-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=8 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-tp --run-name tp-bs8-unpacked
collect_results

log_run tp-bs16-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=16 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-tp --run-name tp-bs16-unpacked
collect_results

log_run tp-bs32-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=32 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-tp --run-name tp-bs32-unpacked
collect_results

log_run tp-bs64-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=64 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-tp --run-name tp-bs64-unpacked
collect_results

log_run tp-bs128-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=128 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-tp --run-name tp-bs128-unpacked
collect_results

#packed
log_run tp-bs2-packed env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=2 MILABENCH_LLM_PACKED=True milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-tp --run-name tp-bs2-packed
collect_results

log_run tp-bs4-packed env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=4 MILABENCH_LLM_PACKED=True milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-tp --run-name tp-bs4-packed
collect_results

log_run tp-bs8-packed env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=8 MILABENCH_LLM_PACKED=True milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-tp --run-name tp-bs8-packed
collect_results


# CP

# unpacked
log_run cp-bs2-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=2 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-cp --run-name cp-bs2-unpacked
collect_results

log_run cp-bs4-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=4 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-cp --run-name cp-bs4-unpacked
collect_results

log_run cp-bs8-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=8 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-cp --run-name cp-bs8-unpacked
collect_results

log_run cp-bs16-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=16 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-cp --run-name cp-bs16-unpacked
collect_results

log_run cp-bs32-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=32 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-cp --run-name cp-bs32-unpacked
collect_results

log_run cp-bs64-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=64 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-cp --run-name cp-bs64-unpacked
collect_results

log_run cp-bs128-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=128 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-cp --run-name cp-bs128-unpacked
collect_results

log_run cp-bs256-unpacked env MILABENCH_SIZER_AUTO=1 MILABENCH_SIZER_BATCH_SIZE=256 MILABENCH_LLM_PACKED=False milabench run --config config/all.yaml --system config/system.yaml --select llm-full-mp-nodes-cp --run-name cp-bs256-unpacked
collect_results


# inference tests
log_run vllm-dense-physics-gpus milabench run --config config/all.yaml --system config/system.yaml --select vllm-dense-physics-gpus
collect_results

log_run vllm-moe-code-gpus milabench run --config config/all.yaml --system config/system.yaml --select vllm-moe-code-gpus
collect_results

# collecting results
# (must run before vllm_main.py below: compare_to_csv.py overwrites results.csv
# from scratch from runs/, which would wipe out vllm_main.py's appended row)
collect_results

# vllm_main.py (standalone custom load test, not a milabench-managed benchmark)
# needs its own vLLM server, since it's just an HTTP client
# same on-disk weights used for the FSDP/TP/CP fine-tuning experiments
# (checkpointer.checkpoint_dir={milabench_data}/llama3_70B there too)
VLLM_MAIN_GPU_COUNT=$(nvidia-smi -L | wc -l)
vllm serve "$MILABENCH_BASE/data/llama3_70B" --served-model-name model --port 8000 --tensor-parallel-size "$VLLM_MAIN_GPU_COUNT" > "$LOG_DIR/vllm-main-server.txt" 2>&1 &
VLLM_MAIN_SERVER_PID=$!

# a 70B model takes a while to load, give it up to 30 minutes to come up
for i in $(seq 1 180); do
  if curl -s -o /dev/null "http://127.0.0.1:8000/health"; then
    break
  fi
  sleep 10
done

log_run vllm-main-n10000-max1000 python vllm_main.py --host 127.0.0.1 --port 8000 --model model -n 10000 --max-tokens 1000 --csv results.csv --run-name vllm-main-n10000-max1000 --bench vllm-main-n10000-max1000

log_run vllm-main-n100-max10000 python vllm_main.py --host 127.0.0.1 --port 8000 --model model -n 100 --max-tokens 10000 --csv results.csv --run-name vllm-main-n100-max10000 --bench vllm-main-n100-max10000

kill "$VLLM_MAIN_SERVER_PID"
wait "$VLLM_MAIN_SERVER_PID" 2>/dev/null
