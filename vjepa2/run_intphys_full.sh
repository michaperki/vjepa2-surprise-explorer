#!/usr/bin/env bash
# Grind the full IntPhys-dev probe to completion through transient WSL2 CUDA
# crashes ("CUDA error: unknown error"). Each crash poisons the CUDA context,
# so recovery requires a fresh process; run_intphys_probe.py resumes from the
# CSV (skips already-scored scenes), so this loop just relaunches until done.
#
#   ./run_intphys_full.sh
#
# Stops when the CSV is complete, or bails if several attempts in a row make no
# progress (which would mean a specific clip reliably crashes, not flakiness).
set -u
cd "$(dirname "$0")"

CSV=outputs/intphys_probe_full.csv
TARGET=361          # 360 movies + 1 header line
MAX_ATTEMPTS=400
STALL_LIMIT=4       # consecutive no-progress attempts before giving up

# Reduce fragmentation-related CUDA errors on long runs.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=/tmp/mpl
export PYTHONPATH=.

# Log everything to a timestamped file so an overnight crash or reboot leaves
# evidence (the terminal scrollback does not survive a WSL2 restart). Both
# stdout and stderr from the whole wrapper are tee'd to this file.
LOG="outputs/intphys_full_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "LOG: $LOG"

attempt=0
prev=0
stall=0
while : ; do
  lines=$(wc -l < "$CSV" 2>/dev/null || echo 0)
  if [ "$lines" -ge "$TARGET" ]; then
    echo "DONE: $CSV has $lines lines (>= $TARGET) after $attempt attempt(s)."
    exit 0
  fi
  if [ "$attempt" -ge 1 ] && [ "$lines" -le "$prev" ]; then
    stall=$((stall + 1))
  else
    stall=0
  fi
  if [ "$stall" -ge "$STALL_LIMIT" ]; then
    echo "STOP: $stall attempts with no new rows at $lines lines."
    echo "      A specific scene is likely crashing every time — inspect the next unscored one."
    exit 2
  fi
  attempt=$((attempt + 1))
  if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
    echo "STOP: hit MAX_ATTEMPTS=$MAX_ATTEMPTS at $lines lines."
    exit 3
  fi
  prev=$lines
  gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null || echo "n/a")
  echo "=== attempt $attempt | CSV $lines/$TARGET lines | GPU free ${gpu_free} | $(date +%H:%M:%S) ==="
  python3 run_intphys_probe.py \
    --source intphys-dev --data-root dev --blocks O1 O2 O3 --max-sets 1000 \
    --weights-dtype bf16 \
    --csv "$CSV" --report REAL_BENCHMARK_FULL.md
  echo "  (probe exited $?)"
done
