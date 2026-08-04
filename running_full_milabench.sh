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

# the live dashboard writes cursor-movement escape codes, which turn into
# unreadable noise once stdout is a file instead of a terminal
export MILABENCH_DASH=no

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

milabench install --config config/all.yaml --system config/system.yaml || exit 1

milabench prepare --config config/all.yaml --system config/system.yaml || exit 1

# 2>&1 matters: tracebacks and milabench's own "Skip <bench> because ..."
# messages go to stderr, so a plain > would silently drop them
milabench run --config config/all.yaml --system config/system.yaml --run-name all > full_milabench_log.txt 2>&1
