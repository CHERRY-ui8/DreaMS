#!/bin/bash
# Watchdog: check if Phase 1 (K-Heads r=256) is done, then launch Phase 2.
# Only runs once — writes a sentinel file to prevent double-launch.

PHASE1_DIR="/root/DreaMS/ms2mol_encdec/outputs/t5_small_phase1_k128_kheads_r256_0521_0311"
SENTINEL="/root/DreaMS/ms2mol_encdec/outputs/_phase2_kheads_launched.flag"

# Phase 1 not done yet
if [ ! -f "$PHASE1_DIR/training.log" ]; then
    exit 0
fi

# Check for "Done!" in training log (Phase 1 completion marker)
if ! grep -q "=== Done" "$PHASE1_DIR/training.log" 2>/dev/null; then
    exit 0  # Phase 1 still running
fi

# Already launched Phase 2
if [ -f "$SENTINEL" ]; then
    exit 0
fi

echo "[_launch_phase2] Phase 1 complete! Starting Phase 2..." >&2

# Launch Phase 2 in background, log to file
cd /root/DreaMS && nohup python -m ms2mol_encdec.train \
  --model_name t5-small --phase 2 --max_epochs 10 \
  --k_tokens 128 \
  --projector_type k_heads --projector_trunk_dim 512 --projector_head_rank 256 \
  --projector_dropout 0.1 --weight_decay 0.1 \
  --batch_size 128 --lr 3e-4 --lr_lora 3e-4 \
  --resume "$PHASE1_DIR/best.ckpt" \
  > /root/DreaMS/ms2mol_encdec/outputs/_phase2_kheads_r256.log 2>&1 &

PH2_PID=$!
echo "$PH2_PID" > "$SENTINEL"
echo "[_launch_phase2] Phase 2 started (PID=$PH2_PID)" >&2
