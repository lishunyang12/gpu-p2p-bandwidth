#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${NVCC:-}" ]]; then
    compiler="$NVCC"
elif [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
    compiler="$CUDA_HOME/bin/nvcc"
else
    compiler="$(command -v nvcc || true)"
fi
if [[ -z "$compiler" ]]; then
    echo 'CUDA Toolkit nvcc was not found. Set CUDA_HOME or NVCC.' >&2
    exit 1
fi

mkdir -p "$repo_dir/.build"
"$compiler" -O3 -std=c++17 "$repo_dir/p2p_bandwidth.cu" -o "$repo_dir/.build/p2p_bandwidth"
if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
    exec "$repo_dir/.build/p2p_bandwidth" --help
fi

result_dir="${OUTPUT_DIR:-$repo_dir/results/$(date -u +%Y%m%dT%H%M%SZ)-$$}"
mkdir -p "$result_dir"
{
    date -u
    uname -sm
    "$compiler" --version
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<all>}"
    printf 'Arguments:'
    printf ' %q' "$@"
    printf '\n'
} > "$result_dir/environment.txt"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi > "$result_dir/nvidia-smi.txt" 2>&1 || true
    nvidia-smi topo -m > "$result_dir/topology.txt" 2>&1 || true
    nvidia-smi topo -p2p r > "$result_dir/p2p-read.txt" 2>&1 || true
    nvidia-smi topo -p2p w > "$result_dir/p2p-write.txt" 2>&1 || true
fi
echo "Results: $result_dir"
"$repo_dir/.build/p2p_bandwidth" --csv "$result_dir/bandwidth.csv" "$@" \
    2>&1 | tee "$result_dir/benchmark.txt"
