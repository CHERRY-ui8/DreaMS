"""Train RAG + T5 for MS → SMILES generation.

The RAG index is built from training embeddings. At each forward pass,
top-3 similar molecules are retrieved and fed as context to T5.

Usage:
    # Phase 1: build index + train projector (freeze T5)
    python /root/DreaMS/rag/train.py --batch_size 32 --max_epochs 5

    # Phase 2: full fine-tune (optional)
    python /root/DreaMS/rag/train.py --batch_size 32 --max_epochs 10 --unfreeze_t5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from ms2mol_rag.model import MSToSMILES_RAG, load_t5_for_rag
from ms2mol_rag.dataset import MSSpectrumSmilesRAGDataset, RAGSmilesCollator
from ms2mol_shared.lora import inject_lora_t5, count_lora_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--k_tokens', type=int, default=16)
    parser.add_argument('--k_contexts', type=int, default=3)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--output_dir', default='/root/DreaMS/ms2mol_rag/outputs')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None)

    # ── Two-phase training ──
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Phase 1=projector only, 2=projector+LoRA')

    # ── LoRA (Phase 2 only) ──
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--lr_lora', type=float, default=3e-4)

    # ── Anti-Lazy-Copying ──
    parser.add_argument('--lazy_penalty', type=str, default='none',
                        choices=['none', 'unlikelihood', 'context_dropout'],
                        help='Anti-lazy-copying mechanism for RAG')
    parser.add_argument('--ul_alpha', type=float, default=0.1,
                        help='Unlikelihood loss weight (only if lazy_penalty=unlikelihood)')
    parser.add_argument('--cd_prob', type=float, default=0.3,
                        help='Context dropout probability (only if lazy_penalty=context_dropout)')

    # ── Instruction Prompt (Phase 2 RAG) ──
    parser.add_argument('--use_instruction', action='store_true', default=False,
                        help='Add fragment-based instruction prompt to encoder input')
    return parser.parse_args()


@torch.no_grad()
def evaluate_generation(model, dataset, device, num_samples=100, num_beams=5):
    """Evaluate generation with RAG retrieval (uses pre-computed neighbors)."""
    model.eval()
    n = min(num_samples, len(dataset))

    all_embs = []
    refs = []
    all_ctx = []
    for i in range(n):
        item = dataset[i]
        all_embs.append(item['ms_emb'])
        refs.append(dataset.smiles[i])
        all_ctx.append(item['neighbors'])

    all_embs = torch.stack(all_embs).to(device)

    batch_size = 64
    all_generated = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            ctx_batch = all_ctx[start:end] if all_ctx[0] is not None else None
            gen = model.generate(
                ms_emb=all_embs[start:end],
                context_smiles=ctx_batch,
                num_beams=num_beams,
                max_length=200,
                device=device,
            )
            all_generated.extend(gen)

    n_valid = 0
    n_exact = 0
    tanimoto_scores = []
    results = []

    for i in range(n):
        ref = refs[i]
        generated = all_generated[i]
        mol_gen = Chem.MolFromSmiles(generated)
        mol_ref = Chem.MolFromSmiles(ref)
        is_valid = mol_gen is not None

        if is_valid:
            n_valid += 1
            gen_canon = Chem.MolToSmiles(mol_gen)
            ref_canon = Chem.MolToSmiles(mol_ref)
            is_exact = gen_canon == ref_canon
            if is_exact:
                n_exact += 1
            fp_gen = AllChem.GetMorganFingerprintAsBitVect(mol_gen, 2, 2048)
            fp_ref = AllChem.GetMorganFingerprintAsBitVect(mol_ref, 2, 2048)
            tan = DataStructs.TanimotoSimilarity(fp_gen, fp_ref)
            tanimoto_scores.append(tan)
        else:
            is_exact = False
            tan = None

        results.append({
            'idx': i, 'reference': ref, 'generated': generated,
            'context': all_ctx[i],
            'valid': is_valid, 'exact_match': is_exact, 'tanimoto': tan,
        })

    metrics = {
        'n_samples': n,
        'valid_rate': round(n_valid / n, 4),
        'exact_match_rate': round(n_exact / n, 6),
        'tanimoto_mean': round(float(np.mean(tanimoto_scores)), 4) if tanimoto_scores else 0.0,
        'tanimoto_median': round(float(np.median(tanimoto_scores)), 4) if tanimoto_scores else 0.0,
    }
    return metrics, results


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f'Device: {device}')
    print(f'Phase: {args.phase}')
    print(f'Architecture: RAG + T5-small (K={args.k_tokens} MS tokens, '
          f'{args.k_contexts} context molecules)')
    if args.lazy_penalty != 'none':
        print(f'Anti-lazy-copying: {args.lazy_penalty}'
              + (f' (alpha={args.ul_alpha})' if args.lazy_penalty == 'unlikelihood' else '')
              + (f' (drop_prob={args.cd_prob})' if args.lazy_penalty == 'context_dropout' else ''))

    # ── Output dir ──
    timestamp = time.strftime('%m%d_%H%M')
    phase_tag = f'phase{args.phase}'
    model_tag = f'k{args.k_tokens}'
    if args.phase == 2:
        model_tag += f'_lora{args.lora_rank}'
    if args.lazy_penalty != 'none':
        model_tag += f'_{args.lazy_penalty}'
    output_dir = os.path.join(args.output_dir, f'rag_{phase_tag}_{model_tag}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Tee stdout
    log_path = os.path.join(output_dir, 'training.log')
    log_file = open(log_path, 'w', buffering=1)
    original_stdout = sys.stdout
    class Tee:
        def write(self, text): original_stdout.write(text); log_file.write(text)
        def flush(self): original_stdout.flush(); log_file.flush()
    sys.stdout = Tee()
    print(f'Output dir: {output_dir}')
    print(f'Log: {log_path}')

    # ── Datasets (neighbors pre-computed in dataset, no KNN at runtime) ──
    print('\n=== Loading datasets ===')
    train_dataset = MSSpectrumSmilesRAGDataset(
        split='train', k_contexts=args.k_contexts,
        cache_dir=output_dir,
    )
    val_dataset = MSSpectrumSmilesRAGDataset(
        split='val', k_contexts=args.k_contexts,
        cache_dir=output_dir,
    )

    # ── Model ──
    print('\n=== Building model ===')
    model = MSToSMILES_RAG(
        k_tokens=args.k_tokens,
        k_contexts=args.k_contexts,
        lazy_penalty=args.lazy_penalty,
        ul_alpha=args.ul_alpha,
        cd_prob=args.cd_prob,
        use_instruction=args.use_instruction,
    ).to(device)
    tokenizer = model.tokenizer

    collator = RAGSmilesCollator(tokenizer, max_length=512)

    # Resume from checkpoint
    if args.resume:
        print(f'Resuming from {args.resume}...')
        state = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(state['model_state_dict'], strict=False)
        if missing:
            print(f'  Missing keys (expected if Phase 1→2): {len(missing)}')
        if unexpected:
            print(f'  Unexpected keys: {len(unexpected)}')

    # Freeze strategy
    if args.phase == 1:
        for p in model.t5.parameters():
            p.requires_grad = False
        print('[Phase 1] T5 backbone: FROZEN')
        print('[Phase 1] Projector: TRAIN')
    else:
        for p in model.t5.parameters():
            p.requires_grad = False
        inject_lora_t5(model.t5, rank=args.lora_rank, alpha=args.lora_alpha)
        print(f'[Phase 2] T5 backbone: FROZEN')
        print(f'[Phase 2] LoRA (rank={args.lora_rank}): TRAIN')
        print('[Phase 2] Projector: TRAIN')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)')

    print(f'\n  Train: {len(train_dataset)} samples ({len(train_loader)} batches)')
    print(f'  Val:   {len(val_dataset)} samples ({len(val_loader)} batches)')

    # ── Optimizer ──
    param_groups = [
        {'params': model.projector.parameters(), 'lr': args.lr},
    ]
    if args.phase == 2:
        lora_params = [p for n, p in model.t5.named_parameters()
                       if 'lora_A' in n or 'lora_B' in n]
        if lora_params:
            param_groups.append({'params': lora_params, 'lr': args.lr_lora})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    total_steps = len(train_loader) * args.max_epochs
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ── Training loop ──
    print(f'\n=== Training ===')
    print(f'Total steps: {total_steps} ({args.max_epochs} epochs × {len(train_loader)} batches)')
    print(f'Warmup: {warmup_steps} steps\n')

    best_val_loss = float('inf')

    for epoch in range(args.max_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        n_batches = 0
        batch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            ms_emb = batch['ms_emb'].to(device)
            labels = batch['labels'].to(device)
            context_smiles = batch.get('context_smiles')

            optimizer.zero_grad()
            outputs = model(
                ms_emb=ms_emb, labels=labels,
                context_smiles=context_smiles,
            )
            loss = outputs['loss']
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - batch_start
                print(f'  Batch {batch_idx+1}/{len(train_loader)} | loss={loss.item():.4f} | {elapsed:.0f}s')
                batch_start = time.time()

        train_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_total = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                ms_emb = batch['ms_emb'].to(device)
                labels = batch['labels'].to(device)
                context_smiles = batch.get('context_smiles')
                outputs = model(
                    ms_emb=ms_emb, labels=labels,
                    context_smiles=context_smiles,
                )
                val_total += outputs['loss'].item()
                val_n += 1

        val_loss = val_total / max(val_n, 1)
        epoch_time = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch+1}/{args.max_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr_now:.2e} | {epoch_time:.0f}s')

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'phase': args.phase,
        }
        torch.save(ckpt, os.path.join(output_dir, 'last.ckpt'))
        if is_best:
            torch.save(ckpt, os.path.join(output_dir, 'best.ckpt'))
            print(f'  ✓ New best val_loss={val_loss:.4f}')

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'  --- Generation eval at epoch {epoch+1} ---')
            eval_metrics, eval_results = evaluate_generation(
                model, val_dataset, device, num_samples=100,
                num_beams=5,
            )
            print(f'  Valid SMILES: {eval_metrics["valid_rate"]*100:.1f}% | '
                  f'Exact match: {eval_metrics["exact_match_rate"]*100:.2f}% | '
                  f'Tanimoto mean: {eval_metrics["tanimoto_mean"]:.4f}')
            eval_path = os.path.join(output_dir, f'eval_epoch{epoch+1}.json')
            with open(eval_path, 'w') as f:
                json.dump({'epoch': epoch + 1, 'metrics': eval_metrics,
                          'results': eval_results}, f, indent=2)
            print(f'  Eval results saved to {eval_path}')

    print('\n=== Done! ===')
    print(f'Best val_loss: {best_val_loss:.4f}')
    print(f'Output: {output_dir}')


if __name__ == '__main__':
    main()
