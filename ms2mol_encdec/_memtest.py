"""Quick memory probe: find max batch_size for K-Heads r=256 full-data Phase 1."""
import subprocess, sys

# Get current memory before (just to confirm GPU is free)
result = subprocess.run(
    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
    capture_output=True, text=True
)
print(f"GPU mem before: {result.stdout.strip()} MiB")

for bs in [64, 128, 192, 256]:
    cmd = (
        f"cd /root/DreaMS && python -m ms2mol_encdec.train "
        f"  --model_name t5-small --phase 1 --max_epochs 1 "
        f"  --k_tokens 128 "
        f"  --projector_type k_heads --projector_trunk_dim 512 --projector_head_rank 256 "
        f"  --projector_dropout 0.1 --weight_decay 0.1 "
        f"  --batch_size {bs} --lr 3e-4 --max_steps 5"
    )
    print(f"\n--- Testing batch_size={bs} ---")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    # Check GPU memory after
    mem_result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    mem_used = int(mem_result.stdout.strip())
    print(f"GPU mem after: {mem_used} MiB")
    
    if result.returncode != 0:
        error = result.stderr[-500:] if result.stderr else result.stdout[-500:]
        if 'out of memory' in error.lower() or 'CUDA' in error or 'memory' in error.lower():
            print(f"  OOM at batch_size={bs}! Max usable: {bs//2}")
            sys.exit(0)
        else:
            print(f"  Other error (returncode={result.returncode}): {error[:200]}")
            sys.exit(1)
    else:
        # Success - check if we're getting close to OOM
        free_mib = 81920 - mem_used
        print(f"  ✓ OK | Used: {mem_used} MiB | Free: {free_mib} MiB")
        if free_mib < 2000:
            print(f"  ← Close to limit, batch_size={bs} is optimal")
            # Don't exit, continue to confirm next size fails
        
print("All batch sizes tested successfully. Max tested: 256")
