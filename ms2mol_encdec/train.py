"""Train multi-backbone T5 (T5 / MolT5 / BioT5) for MS -> SMILES generation.

Unified entry point for both full-dataset training and N-sample overfitting tests.

Usage:
    # Full dataset training
    python -m ms2mol_encdec.train --phase 1 --max_epochs 5 --k_tokens 16

    # N-sample overfitting test (replaces test_overfit_n.py)
    python -m ms2mol_encdec.train --n 5000 --phase 2 --max_epochs 500 --k_tokens 128

    # Phase 2: train projector + LoRA (resume from Phase 1)
    python -m ms2mol_encdec.train --phase 2 --max_epochs 10 --k_tokens 16 \\
        --resume /path/to/phase1/best.ckpt

    # Switch backbone (MolT5-small / BioT5-base / BioT5+ base):
    python -m ms2mol_encdec.train --model_name molt5-small --phase 1 --max_epochs 5
    python -m ms2mol_encdec.train --model_name biot5-base --phase 1 --max_epochs 5

    # Full-sequence: 60 peak positions → T5 (bypasses pooled embedding bottleneck)
    # Requires: extract_embeddings.py with MODE='full' to create full_embedding in HDF5
    python -m ms2mol_encdec.train --projector_type linear_per_peak --phase 1 --max_epochs 5
    python -m ms2mol_encdec.train --projector_type linear_per_peak --phase 2 --max_epochs 10 \\
        --resume /path/to/phase1/best.ckpt
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from ms2mol_encdec.model import MSToSMILES_T5, info_display, MODEL_REGISTRY
from ms2mol_encdec.dataset import (
    MSSpectrumSmilesT5Dataset, MSSpectrumSmilesFullSeqDataset,
    NSubsetDataset, SampledSubsetDataset, T5SmilesCollator,
)
from ms2mol_shared.lora import inject_lora_t5, count_lora_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='t5-small',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Backbone model (t5-small / molt5-small / biot5-base / biot5-plus-base)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--k_tokens', type=int, default=16)
    parser.add_argument('--projector_type', default='mlp',
                        choices=['mlp', 'k_heads', 'qformer', 'linear_per_peak'],
                        help='Projector: mlp (default), k_heads (D1), qformer (learnable queries), '
                             'or linear_per_peak (60 peaks x Linear to d_model)')
    parser.add_argument('--projector_depth', type=int, default=2,
                        help='Depth of projector MLP when projector_type=mlp (2=original)')
    parser.add_argument('--projector_trunk_dim', type=int, default=512,
                        help='Shared trunk width for projector_type=k_heads')
    parser.add_argument('--projector_head_rank', type=int, default=64,
                        help='Bottleneck rank per head for projector_type=k_heads')
    # ── Q-Former args ──
    parser.add_argument('--qformer_num_queries', type=int, default=32,
                        help='Number of learnable queries for projector_type=qformer (default: 32)')
    parser.add_argument('--qformer_layers', type=int, default=4,
                        help='Number of transformer blocks in Q-Former (default: 4)')
    parser.add_argument('--qformer_heads', type=int, default=8,
                        help='Number of attention heads in Q-Former (default: 8)')
    parser.add_argument('--warmup_steps', type=int, default=200)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--output_dir', default='/root/DreaMS/outputs')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint (for Phase 2)')

    # ── Two-phase training ──
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Phase 1=projector only, 2=projector+LoRA')

    # ── LoRA (Phase 2 only) ──
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--lr_lora', type=float, default=3e-4,
                        help='Learning rate for LoRA params')
    parser.add_argument('--projector_lr_scale', type=float, default=0.1,
                        help='Phase 2: scale projector LR = lr * projector_lr_scale '
                             '(default: 0.1). Set 0.0 to freeze projector.')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer (default: 0.01)')
    parser.add_argument('--projector_dropout', type=float, default=0.0,
                        help='Dropout rate for projector MLP (default: 0.0, no dropout)')

    # ── Substructure supervision (Phase 1+: MACCS keys) ──
    parser.add_argument('--maccs_loss_weight', type=float, default=0.0,
                        help='Weight for MACCS BCE loss (default: 0.0 = disabled). '
                             'Set >0 to enable substructure supervision')
    parser.add_argument('--ce_loss_weight', type=float, default=1.0,
                        help='Weight for T5 CE loss (default: 1.0)')

    # ── N-sample overfitting mode ──
    parser.add_argument('--n', type=int, default=0,
                        help='N samples for overfitting test (default: 0 = full dataset). '
                             'Set >0 to run N-sample test (replaces test_overfit_n.py)')
    parser.add_argument('--subset', type=int, default=0,
                        help='Subset size for rapid iteration (default: 0 = full dataset). '
                             'Samples unique molecules, 1 spectrum each. '
                             'E.g. --subset 40000 for ~5 min/epoch.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for N-sample subset selection')
    parser.add_argument('--eval_every', type=int, default=5,
                        help='Generation evaluation every N epochs (default: 5)')
    parser.add_argument('--shuffle_ms', action='store_true',
                        help='Shuffle MS embeddings across batch (mismatch spectra vs labels). '
                             'Diagnostic: if loss/generation don\'t change, decoder ignores encoder input.')
    return parser.parse_args()


@torch.no_grad()
def evaluate_generation(model, dataset, device, num_samples=100, num_beams=5):
    """Run generation evaluation: valid rate, exact match, Tanimoto."""
    model.eval()
    n = min(num_samples, len(dataset))

    all_embs = []
    refs = []
    for i in range(n):
        item = dataset[i]
        all_embs.append(item['ms_emb'])
        refs.append(dataset.smiles[i])
    all_embs = torch.stack(all_embs).to(device)

    batch_size = 64
    all_generated = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            gen = model.generate(
                ms_emb=all_embs[start:end],
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
    display = info_display(args.model_name)
    print(f'Device: {device}')
    print(f'Phase: {args.phase}')
    print(f'Model: {display} (--model_name={args.model_name})')
    print(f'Architecture: backbone + projector (K={args.k_tokens if args.projector_type != "linear_per_peak" else 60} prefix tokens)')
    if args.n > 0:
        print(f'Mode: N-sample overfitting (N={args.n}, seed={args.seed})')
    elif args.subset > 0:
        print(f'Mode: Subset iteration ({args.subset} unique molecules, seed={args.seed})')
    else:
        print(f'Mode: Full dataset training')

    # ── Model ──
    print('\n=== Building model ===')
    model = MSToSMILES_T5(
        k_tokens=args.k_tokens, model_name=args.model_name,
        projector_type=args.projector_type,
        projector_depth=args.projector_depth,
        projector_dropout=args.projector_dropout,
        projector_trunk_dim=args.projector_trunk_dim,
        projector_head_rank=args.projector_head_rank,
        qformer_num_queries=args.qformer_num_queries,
        qformer_layers=args.qformer_layers,
        qformer_heads=args.qformer_heads,
        maccs_loss_weight=args.maccs_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
    ).to(device)
    tokenizer = model.tokenizer

    # ── Phase setup ──
    if args.phase == 1:
        # Phase 1: freeze backbone, train only projector
        for p in model.t5.parameters():
            p.requires_grad = False
        print(f'[Phase 1] {display} backbone: FROZEN')
        print('[Phase 1] Projector: TRAIN')
        if hasattr(model, 'maccs_head') and model.maccs_head is not None:
            print(f'[Phase 1] MACCS substructure head: TRAIN (β={args.maccs_loss_weight})')
    else:
        # Phase 2: freeze backbone, inject LoRA, train projector + LoRA
        for p in model.t5.parameters():
            p.requires_grad = False
        inject_lora_t5(model.t5, rank=args.lora_rank, alpha=args.lora_alpha)
        print(f'[Phase 2] {display} backbone: FROZEN')
        print(f'[Phase 2] LoRA (rank={args.lora_rank}): TRAIN')
        print('[Phase 2] Projector: TRAIN')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)')

    # ── Resume from checkpoint ──
    start_epoch = 0
    if args.resume:
        print(f'Resuming from {args.resume}...')
        state = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(state['model_state_dict'], strict=False)
        if missing:
            print(f'  Missing keys (expected if Phase 1->2): {len(missing)}')
        if unexpected:
            print(f'  Unexpected keys: {len(unexpected)}')
        if 'epoch' in state:
            start_epoch = state['epoch']
            print(f'  Resuming from epoch {start_epoch}')

    # ── Datasets ──
    print('\n=== Loading datasets ===')
    is_nmode = args.n > 0
    is_subset_mode = args.subset > 0

    # linear_per_peak requires MSSpectrumSmilesFullSeqDataset (full 60x1024 sequence)
    if args.projector_type == 'linear_per_peak':
        if is_subset_mode:
            print(f'  Using sampled subset: {args.subset} unique molecules (full_embedding, lazy)')
            train_dataset = SampledSubsetDataset(
                split='train', n=args.subset, seed=args.seed,
                embedding_key='full_embedding',
            )
            val_dataset = SampledSubsetDataset(
                split='val', n=max(2000, args.subset // 20), seed=args.seed,
                embedding_key='full_embedding',
            )
        elif is_nmode:
            print('  [linear_per_peak] N-mode uses full dataset; set --n=0 for subset control')
            train_dataset = MSSpectrumSmilesFullSeqDataset(split='train')
            val_dataset = MSSpectrumSmilesFullSeqDataset(split='val')
        else:
            train_dataset = MSSpectrumSmilesFullSeqDataset(split='train')
            val_dataset = MSSpectrumSmilesFullSeqDataset(split='val')
    elif is_subset_mode:
        train_dataset = SampledSubsetDataset(split='train', n=args.subset, seed=args.seed)
        val_dataset = SampledSubsetDataset(split='val', n=max(2000, args.subset // 20), seed=args.seed)
    elif is_nmode:
        train_dataset = NSubsetDataset(n=args.n, seed=args.seed)
        val_dataset = None
    else:
        train_dataset = MSSpectrumSmilesT5Dataset(split='train')
        val_dataset = MSSpectrumSmilesT5Dataset(split='val')

    collator = T5SmilesCollator(tokenizer, max_length=512)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=0,
    )
    print(f'\n  Train: {len(train_dataset)} samples ({len(train_loader)} batches)')

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collator, num_workers=0,
        )
        print(f'  Val:   {len(val_dataset)} samples ({len(val_loader)} batches)')
    else:
        # N-mode: no separate validation set
        val_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collator, num_workers=0,
        )

    # ── Optimizer ──
    projector_lr = args.lr
    if args.phase == 2 and args.projector_lr_scale > 0:
        projector_lr = args.lr * args.projector_lr_scale
        print(f'[Optimizer] Phase 2: projector LR = {projector_lr:.2e} '
              f'(lr={args.lr:.2e} × scale={args.projector_lr_scale})')
    elif args.phase == 2 and args.projector_lr_scale == 0:
        for p in model.projector.parameters():
            p.requires_grad = False
        print('[Optimizer] Phase 2: projector FROZEN (projector_lr_scale=0)')

    param_groups = [
        {'params': model.projector.parameters(), 'lr': projector_lr},
    ]
    if hasattr(model, 'maccs_head') and model.maccs_head is not None:
        param_groups[0]['params'] = list(model.projector.parameters()) + \
                                     list(model.maccs_head.parameters())
        print(f'[Optimizer] MACCS head params added to projector group')
    if args.phase == 2:
        lora_params = [p for n, p in model.t5.named_parameters()
                       if 'lora_A' in n or 'lora_B' in n]
        if lora_params:
            param_groups.append({'params': lora_params, 'lr': args.lr_lora})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.max_epochs
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ── Output dir ──
    timestamp = time.strftime('%m%d_%H%M')
    phase_tag = f'phase{args.phase}'
    if args.projector_type == 'k_heads':
        arch_tag = f'k{args.k_tokens}_kheads_r{args.projector_head_rank}'
    elif args.projector_type == 'qformer':
        arch_tag = f'q{args.qformer_num_queries}_l{args.qformer_layers}'
    elif args.projector_type == 'linear_per_peak':
        arch_tag = 'fullseq_linear'
    else:
        arch_tag = f'k{args.k_tokens}_d{args.projector_depth}'

    # Data tag
    if args.subset > 0:
        data_tag = f's{args.subset}'
    elif args.n > 0:
        data_tag = f'n{args.n}'
    else:
        data_tag = 'full'

    # Init tag
    init_tag = 'resume' if args.resume else 'scratch'

    # MACCS tag
    maccs_tag = f'maccs{args.maccs_loss_weight}' if args.maccs_loss_weight > 0 else 'nomaccs'

    # Phase 2 specific tag
    phase2_tag = ''
    if args.phase == 2:
        proj_tag = 'uproj' if args.projector_lr_scale > 0 else 'fproj'
        phase2_tag = f'_lora{args.lora_rank}_{proj_tag}'

    output_dir = os.path.join(
        args.output_dir,
        f'{phase_tag}_{arch_tag}_{data_tag}_{init_tag}_{maccs_tag}{phase2_tag}_{timestamp}'
    )
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, 'training.log')
    log_file = open(log_path, 'w', buffering=1)
    original_stdout = sys.stdout

    class Tee:
        def write(self, text): original_stdout.write(text); log_file.write(text)
        def flush(self): original_stdout.flush(); log_file.flush()
    sys.stdout = Tee()
    print(f'Output dir: {output_dir}')
    print(f'Log: {log_path}')

    # ── Training loop ──
    print(f'\n=== Training ===')
    print(f'Total steps: {total_steps} ({args.max_epochs} epochs x {len(train_loader)} batches)')
    print(f'Warmup: {warmup_steps} steps\n')

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.max_epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        total_ce = 0
        total_bce = 0
        total_maccs_correct = 0
        total_maccs_bits = 0
        n_batches = 0
        batch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            ms_emb = batch['ms_emb'].to(device)
            labels = batch['labels'].to(device)
            maccs = batch.get('maccs', None)
            if maccs is not None:
                maccs = maccs.to(device)

            optimizer.zero_grad()
            outputs = model(ms_emb=ms_emb, labels=labels, maccs=maccs)
            loss = outputs['loss']
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            # Track MACCS metrics
            if outputs.get('maccs_logits') is not None and maccs is not None:
                total_ce += outputs.get('ce_loss', loss).item()
                bc = outputs['maccs_loss']
                if bc is not None:
                    total_bce += bc.item()
                    preds = (outputs['maccs_logits'] > 0).float()
                    total_maccs_correct += (preds == maccs).sum().item()
                    total_maccs_bits += maccs.numel()

            if (batch_idx + 1) % 100 == 0:
                elapsed = time.time() - batch_start
                print(f'  Batch {batch_idx+1}/{len(train_loader)} | loss={loss.item():.4f} | {elapsed:.0f}s')
                batch_start = time.time()

        train_loss = total_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_total = 0
        val_maccs_correct = 0
        val_maccs_bits = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                ms_emb = batch['ms_emb'].to(device)
                labels = batch['labels'].to(device)
                maccs = batch.get('maccs', None)
                if maccs is not None:
                    maccs = maccs.to(device)
                outputs = model(ms_emb=ms_emb, labels=labels, maccs=maccs)
                val_total += outputs['loss'].item()
                val_n += 1
                # Track val MACCS accuracy
                if outputs.get('maccs_logits') is not None and maccs is not None:
                    preds = (outputs['maccs_logits'] > 0).float()
                    val_maccs_correct += (preds == maccs).sum().item()
                    val_maccs_bits += maccs.numel()

        val_loss = val_total / max(val_n, 1)
        epoch_time = time.time() - epoch_start
        lr_now = optimizer.param_groups[0]['lr']

        # Build epoch log line
        log_line = (f'Epoch {epoch+1}/{args.max_epochs} | '
                    f'train_loss={train_loss:.4f} | val_loss={val_loss:.4f}')
        if total_maccs_bits > 0:
            train_maccs_acc = total_maccs_correct / total_maccs_bits * 100
            val_maccs_acc = val_maccs_correct / max(val_maccs_bits, 1) * 100
            avg_bce = total_bce / max(n_batches, 1)
            avg_ce = total_ce / max(n_batches, 1)
            log_line += (f' | ce={avg_ce:.4f} bce={avg_bce:.4f}'
                         f' | MACCS acc train={train_maccs_acc:.1f}% val={val_maccs_acc:.1f}%')
        log_line += f' | lr={lr_now:.2e} | {epoch_time:.0f}s'
        print(log_line)

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
            'model_name': args.model_name,
            'k_tokens': args.k_tokens,
            'projector_type': args.projector_type,
            'projector_depth': args.projector_depth,
            'projector_trunk_dim': args.projector_trunk_dim,
            'projector_head_rank': args.projector_head_rank,
            'projector_dropout': args.projector_dropout,
            'maccs_loss_weight': args.maccs_loss_weight,
            'ce_loss_weight': args.ce_loss_weight,
        }
        torch.save(ckpt, os.path.join(output_dir, 'last.ckpt'))
        if is_best:
            torch.save(ckpt, os.path.join(output_dir, 'best.ckpt'))
            print(f'  ✓ New best val_loss={val_loss:.4f}')

        if (epoch + 1) % args.eval_every == 0:
            print(f'  --- Generation eval at epoch {epoch+1} ---')
            eval_dataset = train_dataset if is_nmode else val_dataset
            eval_metrics, eval_results = evaluate_generation(
                model, eval_dataset, device, num_samples=min(100, len(eval_dataset)), num_beams=5,
            )
            print(f'  Valid SMILES: {eval_metrics["valid_rate"]*100:.1f}% | '
                  f'Exact match: {eval_metrics["exact_match_rate"]*100:.2f}% | '
                  f'Tanimoto mean: {eval_metrics["tanimoto_mean"]:.4f}')
            eval_path = os.path.join(output_dir, f'eval_epoch{epoch+1}.json')
            with open(eval_path, 'w') as f:
                json.dump({'epoch': epoch + 1, 'metrics': eval_metrics, 'results': eval_results}, f, indent=2)
            print(f'  Eval results saved to {eval_path}')

    print('\n=== Done! ===')
    print(f'Best val_loss: {best_val_loss:.4f}')
    print(f'Output: {output_dir}')


if __name__ == '__main__':
    main()
