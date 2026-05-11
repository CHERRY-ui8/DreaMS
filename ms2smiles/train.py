"""Train MS→SMILES model with prefix tuning.

Usage:
    # Phase 1: freeze ChemGPT backbone, train projector + embedding + lm_head
    conda run -n dreams python -m ms2smiles.train --model_size 19M --max_epochs 50

    # Phase 2: full fine-tune
    conda run -n dreams python -m ms2smiles.train --model_size 19M --max_epochs 30 \\
        --freeze_chemgpt_backbone false --lr_backbone 1e-5
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from ms2smiles.config import MS2SMILESConfig
from ms2smiles.model import MStoSMILES
from ms2smiles.dataset import MSSpectrumSmilesDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', default='19M', choices=['19M', '1.2B'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--lr_projector', type=float, default=3e-4)
    parser.add_argument('--lr_emb', type=float, default=3e-5)
    parser.add_argument('--lr_backbone', type=float, default=1e-5)
    parser.add_argument('--freeze_chemgpt_backbone', action='store_true', default=True)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--output_dir', default='/root/DreaMS/ms2smiles/outputs')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint to resume from')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def freeze_strategy(model: MStoSMILES, freeze_backbone: bool):
    """Apply freeze strategy.

    Phase 1 (freeze_backbone=True):
        - Projector: TRAIN
        - ChemGPT LM head: TRAIN
        - ChemGPT token embedding (wte): TRAIN
        - ChemGPT transformer backbone: FROZEN

    Phase 2 (freeze_backbone=False):
        - Everything: TRAIN
    """
    if not freeze_backbone:
        for p in model.chemgpt.parameters():
            p.requires_grad = True
        print('[Freeze] All ChemGPT params: TRAIN')
        return

    # Freeze all ChemGPT first
    for p in model.chemgpt.parameters():
        p.requires_grad = False

    # Unfreeze lm_head and token embeddings
    for name, p in model.chemgpt.named_parameters():
        if 'lm_head' in name or name == 'transformer.wte.weight':
            p.requires_grad = True

    # Projector is always trainable (set in __init__)
    print('[Freeze] ChemGPT backbone: FROZEN')
    print('[Freeze] ChemGPT lm_head + wte: TRAIN')
    print('[Freeze] Projector: TRAIN')


def get_param_groups(model: MStoSMILES, args):
    """Create optimizer param groups with different LRs."""
    groups = [
        {'params': model.projector.parameters(), 'lr': args.lr_projector},
        # lm_head.weight IS wte.weight (weight tying) — only add once
        {'params': [model.chemgpt.lm_head.weight], 'lr': args.lr_emb},
    ]

    if hasattr(model.chemgpt.lm_head, 'bias') and model.chemgpt.lm_head.bias is not None:
        groups.append({'params': [model.chemgpt.lm_head.bias], 'lr': args.lr_emb})

    if not args.freeze_chemgpt_backbone:
        # Add backbone params with lower LR
        backbone_params = []
        for name, p in model.chemgpt.named_parameters():
            if 'lm_head' not in name and name != 'transformer.wte.weight':
                backbone_params.append(p)
        groups.append({'params': backbone_params, 'lr': args.lr_backbone})

    return groups


def train_epoch(model, loader, optimizer, scheduler, device, grad_clip):
    model.train()
    total_loss = 0
    n_batches = 0
    start = time.time()

    for batch_idx, batch in enumerate(loader):
        # Move to device
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
def validate(model, loader, device):
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
        total_loss += outputs['loss'].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f'Device: {device}')

    # Config
    config = MS2SMILESConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr_projector=args.lr_projector,
        lr_chemgpt_emb=args.lr_emb,
        lr_chemgpt_backbone=args.lr_backbone,
        freeze_chemgpt_backbone=args.freeze_chemgpt_backbone,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
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

    if args.resume:
        print(f'Resuming from {args.resume}...')
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        # Also need to load training state...

    # Freeze strategy
    print('\n=== Freeze strategy ===')
    freeze_strategy(model, args.freeze_chemgpt_backbone)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)')

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

    # Resume optimizer state (if checkpoint provided)
    start_epoch = 0
    best_val_loss = float('inf')

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    print(f'\n=== Training ===')
    print(f'Total steps: {total_steps} ({args.max_epochs} epochs × {len(train_loader)} batches)')
    print(f'Warmup: {warmup_steps} steps')
    print()

    for epoch in range(start_epoch, args.max_epochs):
        epoch_start = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.grad_clip,
        )
        val_loss = validate(model, val_loader, device)
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
        torch.save(ckpt, os.path.join(args.output_dir, 'last.ckpt'))

        # Save best
        if is_best:
            torch.save(ckpt, os.path.join(args.output_dir, 'best.ckpt'))
            print(f'  ✓ New best val_loss={val_loss:.4f}')

        # Generate sample SMILES from val set every 5 epochs
        if (epoch + 1) % 5 == 0:
            sample_idx = 0
            sample_emb = val_dataset[sample_idx]['embedding'].unsqueeze(0).to(device)
            sample_smi = val_dataset.smiles[sample_idx]
            gen_smi = model.generate(embeddings=sample_emb, num_beams=3)[0]
            print(f'  Sample: truth="{sample_smi}"')
            print(f'          pred="{gen_smi}"')

    print('\n=== Done! ===')


if __name__ == '__main__':
    main()
