"""Train MS→SMILES model with multi-token prefix + LoRA + two-phase training.

Usage:
    # Phase 1: train projector only (alignment, 3-5 epochs)
    python -m ms2smiles.train --model_size 19M --phase 1 --max_epochs 5

    # Phase 2: train projector + LoRA (further tuning, 30 epochs)
    python -m ms2smiles.train --model_size 19M --phase 2 --max_epochs 30
        --resume /path/to/phase1/best.ckpt
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import selfies as sf

from ms2smiles.config import MS2SMILESConfig
from ms2smiles.model import MStoSMILES, inject_lora, count_lora_params
from ms2smiles.dataset import MSSpectrumSmilesDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', default='19M', choices=['19M', '1.2B'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--lr_projector', type=float, default=3e-4)
    parser.add_argument('--lr_emb', type=float, default=3e-5)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--lr_lora', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--output_dir', default='/root/DreaMS/ms2smiles/outputs')

    # ── Multi-token prefix ──
    parser.add_argument('--k_tokens', type=int, default=4,
                        help='Number of prefix tokens for MS conditioning')

    # ── LoRA ──
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--lora_target', type=str, default='q_proj,v_proj')

    # ── Training phase ──
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Phase 1=projector only, 2=projector+LoRA')

    # ── Loss reweighting ──
    parser.add_argument('--loss_reweight', action='store_true', default=True)
    parser.add_argument('--no-loss_reweight', action='store_false', dest='loss_reweight')
    parser.add_argument('--loss_reweight_values', type=float, nargs='+',
                        default=[10.0, 8.0, 5.0, 2.0])

    parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume from')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def freeze_strategy_phase1(model: MStoSMILES):
    """Phase 1: train ONLY projector. Everything else frozen."""
    # Freeze everything in ChemGPT
    for p in model.chemgpt.parameters():
        p.requires_grad = False

    # Projector is always trainable (set in __init__)
    print('[Freeze Phase 1] ChemGPT (all): FROZEN')
    print('[Freeze Phase 1] Projector: TRAIN')


def freeze_strategy_phase2(model: MStoSMILES, config: MS2SMILESConfig):
    """Phase 2: train projector + LoRA. ChemGPT backbone frozen."""
    # Freeze ChemGPT backbone
    for p in model.chemgpt.parameters():
        p.requires_grad = False

    # Unfreeze LoRA parameters
    for n, p in model.chemgpt.named_parameters():
        if 'lora_A' in n or 'lora_B' in n:
            p.requires_grad = True

    # Projector is always trainable
    print('[Freeze Phase 2] ChemGPT backbone: FROZEN')
    print('[Freeze Phase 2] LoRA params: TRAIN')
    print(f'[Freeze Phase 2] Projector: TRAIN')


def get_param_groups(model: MStoSMILES, args):
    """Create optimizer param groups with different LRs."""
    groups = [
        {'params': model.projector.parameters(), 'lr': args.lr_projector},
    ]

    if args.phase == 2:
        # LoRA params with higher LR
        lora_params = [p for n, p in model.chemgpt.named_parameters()
                       if 'lora_A' in n or 'lora_B' in n]
        if lora_params:
            groups.append({'params': lora_params, 'lr': args.lr_lora})

    return groups


def compute_weighted_loss(logits, labels, k_tokens, reweight_values):
    """Compute time-step weighted cross-entropy loss.

    Logits are from model (before HF's internal loss computation).
    We override the loss with our own weighted version.

    Args:
        logits: (B, K+S, V) raw model logits
        labels: (B, K+S) labels with -100 for prefix positions
        k_tokens: number of prefix tokens
        reweight_values: list/tuple of [w1, w2, ...] for first N real token predictions
    Returns:
        scalar loss
    """
    # Apply HF-style shift: predict next token
    shift_logits = logits[:, :-1, :].contiguous()  # (B, K+S-1, V)
    shift_labels = labels[:, 1:].contiguous()       # (B, K+S-1)

    loss_fct = nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1))
    loss = loss.view(shift_logits.size(0), shift_logits.size(1))  # (B, K+S-1)

    # Build weight matrix
    weights = torch.ones_like(loss)
    start_idx = k_tokens - 1  # first position that predicts a real token

    for i, w in enumerate(reweight_values):
        pos = start_idx + i
        if pos < loss.size(1):
            weights[:, pos] = w

    # Weighted sum over non-padded positions
    mask = (shift_labels != -100).float()
    weighted_loss = (loss * weights * mask).sum() / mask.sum().clamp(min=1)
    return weighted_loss


def train_epoch(model, loader, optimizer, scheduler, device, grad_clip, args):
    model.train()
    total_loss = 0
    n_batches = 0
    start = time.time()

    for batch_idx, batch in enumerate(loader):
        smiles_ids = batch['smiles_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        embeddings = batch['embeddings'].to(device)

        optimizer.zero_grad()
        outputs = model(
            smiles_ids=smiles_ids,
            labels=labels,
            attention_mask=attention_mask,
            embeddings=embeddings,
        )

        if args.loss_reweight:
            # Override loss with time-step weighted version
            loss = compute_weighted_loss(
                outputs['logits'],
                outputs['labels'],  # includes prefix -100 labels
                k_tokens=args.k_tokens,
                reweight_values=args.loss_reweight_values,
            )
        else:
            loss = outputs['loss']

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f'  Batch {batch_idx+1}/{len(loader)} | loss={loss.item():.4f} | '
                  f'{elapsed:.0f}s')
            start = time.time()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    total_loss = 0
    n_batches = 0

    for batch in loader:
        smiles_ids = batch['smiles_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        embeddings = batch['embeddings'].to(device)

        outputs = model(
            smiles_ids=smiles_ids,
            labels=labels,
            attention_mask=attention_mask,
            embeddings=embeddings,
        )

        if args.loss_reweight:
            loss = compute_weighted_loss(
                outputs['logits'],
                outputs['labels'],
                k_tokens=args.k_tokens,
                reweight_values=args.loss_reweight_values,
            )
        else:
            loss = outputs['loss']

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_generation(model, dataset, device, num_samples=100, num_beams=5):
    """Run full generation evaluation: valid rate, exact match, Tanimoto."""
    model.eval()
    n = min(num_samples, len(dataset))

    # Collect embeddings and references
    all_embs = []
    refs = []
    for i in range(n):
        item = dataset[i]
        all_embs.append(item['embedding'])
        refs.append(dataset.smiles[i])
    all_embs = torch.stack(all_embs).to(device)

    # Generate in batches
    batch_size = 64
    all_generated = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            gen = model.generate(
                embeddings=all_embs[start:end],
                num_beams=num_beams,
                max_length=200,
            )
            all_generated.extend(gen)

    # Evaluate
    n_valid = 0
    n_exact = 0
    tanimoto_scores = []
    results = []

    for i in range(n):
        ref = refs[i]
        raw = all_generated[i]
        # Remove spaces from SELFIES
        clean = raw.replace(' ', '')
        try:
            gen_smi = sf.decoder(clean)
        except Exception:
            gen_smi = None

        if gen_smi is None:
            results.append({'idx': i, 'reference': ref, 'generated': raw,
                           'valid': False, 'exact_match': False, 'tanimoto': None})
            continue

        mol_gen = Chem.MolFromSmiles(gen_smi)
        mol_ref = Chem.MolFromSmiles(ref)
        is_valid = mol_gen is not None

        if is_valid:
            n_valid += 1
            gen_canon = Chem.MolToSmiles(mol_gen)
            ref_canon = Chem.MolToSmiles(mol_ref)
            is_exact = gen_canon == ref_canon
            if is_exact:
                n_exact += 1
            # Tanimoto
            fp_gen = AllChem.GetMorganFingerprintAsBitVect(mol_gen, 2, 2048)
            fp_ref = AllChem.GetMorganFingerprintAsBitVect(mol_ref, 2, 2048)
            tan = DataStructs.TanimotoSimilarity(fp_gen, fp_ref)
            tanimoto_scores.append(tan)
        else:
            is_exact = False
            tan = None

        results.append({
            'idx': i, 'reference': ref, 'generated': gen_smi,
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
    print(f'K tokens: {args.k_tokens}')
    if args.phase == 2:
        print(f'LoRA rank={args.lora_rank}, alpha={args.lora_alpha}, targets={args.lora_target}')

    # Config
    config = MS2SMILESConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr_projector=args.lr_projector,
        lr_chemgpt_emb=args.lr_emb,
        lr_chemgpt_backbone=args.lr_backbone,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        k_tokens=args.k_tokens,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target,
        phase=args.phase,
        loss_reweight=args.loss_reweight,
        loss_reweight_values=tuple(args.loss_reweight_values),
    )

    # Datasets
    print('\n=== Loading datasets ===')
    train_dataset = MSSpectrumSmilesDataset(config, split='train')
    val_dataset = MSSpectrumSmilesDataset(config, split='val')
    test_dataset = MSSpectrumSmilesDataset(config, split='test')

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f'\n  Train: {len(train_dataset)} samples ({len(train_loader)} batches)')
    print(f'  Val:   {len(val_dataset)} samples ({len(val_loader)} batches)')
    print(f'  Test:  {len(test_dataset)} samples')

    # Model
    print('\n=== Building model ===')
    model = MStoSMILES(config, device=device)

    # Inject LoRA for Phase 2
    if args.phase == 2:
        target_list = [t.strip() for t in args.lora_target.split(',')]
        inject_lora(model.chemgpt, rank=args.lora_rank, alpha=args.lora_alpha,
                    target_modules=target_list)

    # Resume from checkpoint
    if args.resume:
        print(f'Resuming from {args.resume}...')
        state = torch.load(args.resume, map_location=device)
        # If resuming, also handle LoRA weights
        missing, unexpected = model.load_state_dict(state['model_state_dict'], strict=False)
        if missing:
            print(f'  Missing keys in checkpoint (expected if Phase 1->2): {len(missing)}')
        if unexpected:
            print(f'  Unexpected keys in checkpoint: {len(unexpected)}')

    # Freeze strategy
    print('\n=== Freeze strategy ===')
    if args.phase == 1:
        freeze_strategy_phase1(model)
    else:
        freeze_strategy_phase2(model, config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)')

    # Optimizer
    param_groups = get_param_groups(model, args)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    # Scheduler: linear warmup then cosine
    total_steps = len(train_loader) * args.max_epochs
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # Output dir
    timestamp = time.strftime('%m%d_%H%M')
    phase_tag = f'phase{args.phase}'
    model_tag = f'k{args.k_tokens}'
    if args.phase == 2:
        model_tag += f'_lora{args.lora_rank}'
    output_dir = os.path.join(args.output_dir, f'{phase_tag}_{model_tag}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'\nOutput dir: {output_dir}')

    # Training loop
    print(f'\n=== Training ===')
    print(f'Total steps: {total_steps} ({args.max_epochs} epochs × {len(train_loader)} batches)')
    print(f'Warmup: {warmup_steps} steps')
    print()

    start_epoch = 0
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.max_epochs):
        epoch_start = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.grad_clip, args,
        )
        val_loss = validate(model, val_loader, device, args)
        epoch_time = time.time() - epoch_start

        lr_now = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch+1}/{args.max_epochs} | '
              f'train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | '
              f'lr={lr_now:.2e} | {epoch_time:.0f}s')

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
            'config': config,
        }

        # Save latest
        torch.save(ckpt, os.path.join(output_dir, 'last.ckpt'))

        # Save best
        if is_best:
            torch.save(ckpt, os.path.join(output_dir, 'best.ckpt'))
            print(f'  ✓ New best val_loss={val_loss:.4f}')

        # Evaluate generation every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'  --- Generation eval at epoch {epoch+1} ---')
            eval_metrics, eval_results = evaluate_generation(
                model, val_dataset, device, num_samples=100, num_beams=5,
            )
            print(f'  Valid SMILES: {eval_metrics["valid_rate"]*100:.1f}% | '
                  f'Exact match: {eval_metrics["exact_match_rate"]*100:.2f}% | '
                  f'Tanimoto mean: {eval_metrics["tanimoto_mean"]:.4f}')
            # Save eval results
            eval_path = os.path.join(output_dir, f'eval_epoch{epoch+1}.json')
            with open(eval_path, 'w') as f:
                json.dump({'epoch': epoch + 1, 'metrics': eval_metrics, 'results': eval_results},
                         f, indent=2)
            print(f'  Eval results saved to {eval_path}')

    print('\n=== Done! ===')
    print(f'Best val_loss: {best_val_loss:.4f}')
    print(f'Output saved to: {output_dir}')


if __name__ == '__main__':
    main()
