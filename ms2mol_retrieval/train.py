"""Train CLIP-style retrieval model for MS ↔ molecule contrastive learning.

Usage:
    # Full training
    python -m ms2mol_retrieval.train \
        --batch_size 256 --max_epochs 50 --lr 3e-4 \
        --proj_dim 256

    # Quick smoke test
    python -m ms2mol_retrieval.train \
        --batch_size 32 --max_epochs 3 \
        --subset 2000 --no_faiss_eval
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from ms2mol_retrieval.model import MSMolCLIP
from ms2mol_retrieval.dataset import MSMolRetrievalDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train CLIP-style MS ↔ molecule retrieval model',
    )

    # Data
    parser.add_argument('--hdf5_path', default='/root/datasets/pairs_with_embs.hdf5')
    parser.add_argument('--subset', type=int, default=None,
                        help='Use only N training samples (for quick iteration)')
    parser.add_argument('--force_molformer_recompute', action='store_true',
                        help='Recompute MoLFormer embeddings even if cached')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Shared cache dir for MoLFormer embeddings (default: output_dir)')

    # Architecture
    parser.add_argument('--ms_dim', type=int, default=1024)
    parser.add_argument('--mol_dim', type=int, default=768)
    parser.add_argument('--proj_dim', type=int, default=256,
                        help='Shared projection dimension')
    parser.add_argument('--proj_hidden', type=int, default=1024,
                        help='Projector hidden layer width')
    parser.add_argument('--proj_depth', type=int, default=2, choices=[2, 3],
                        help='Projector depth (2=one hidden layer, 3=two hidden layers)')

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--eval_every', type=int, default=5,
                        help='Run FAISS full-database eval every N epochs')
    parser.add_argument('--no_faiss_eval', action='store_true',
                        help='Skip FAISS full-database evaluation')
    parser.add_argument('--num_workers', type=int, default=0)

    # Hard negative mining
    parser.add_argument('--hard_ratio', type=float, default=0.0,
                        help='Fraction of batch from same molecular cluster '
                             '(0.0 = standard random sampling). E.g. 0.25 = '
                             '25% hard negatives from same cluster + 75% random)')
    parser.add_argument('--hard_ratio_phase2', type=float, default=None,
                        help='Phase 2 hard_ratio for curriculum learning. '
                             'When set, switches from hard_ratio=0.0 to this '
                             'value at --switch_epoch.')
    parser.add_argument('--switch_epoch', type=int, default=10,
                        help='Epoch to switch from phase 1 to phase 2 '
                             '(default: 10). Only used with --hard_ratio_phase2.')

    # Output
    parser.add_argument('--output_dir', default='/root/DreaMS/ms2mol_retrieval/outputs')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)

    # Wandb
    parser.add_argument('--no_wandb', action='store_true', default=True,
                        help='Disable wandb (default: True for local runs)')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='ms2mol_retrieval')

    return parser.parse_args()


def build_val_index(
    model: MSMolCLIP,
    val_dataset: MSMolRetrievalDataset,
    device: str = 'cuda',
    batch_size: int = 512,
) -> dict:
    """Encode full validation set and compute retrieval metrics.

    Returns dict with recall@{1,5,10}.
    """
    model.eval()
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)

    all_ms = []
    all_mol = []
    all_mol_ids = []

    with torch.no_grad():
        for batch in loader:
            ms = batch['ms_emb'].to(device)
            mol = batch['mol_emb'].to(device)
            ms_feat = model.encode_ms(ms)
            mol_feat = model.encode_mol(mol)
            all_ms.append(ms_feat.cpu())
            all_mol.append(mol_feat.cpu())
            all_mol_ids.extend(batch['mol_id'].numpy().tolist())

    ms_feat_all = torch.cat(all_ms, dim=0)  # (N_val, proj_dim)
    mol_feat_all = torch.cat(all_mol, dim=0)  # (N_val, proj_dim)

    # Deduplicate molecule embeddings (unique molecules)
    unique_ids = sorted(set(all_mol_ids))
    id_to_unique = {uid: i for i, uid in enumerate(unique_ids)}
    mol_labels_unique = list(unique_ids)
    ms_labels = all_mol_ids  # Each MS query's correct molecule

    # Average molecule embeddings for molecules with multiple spectra
    mol_feat_unique = torch.zeros(len(unique_ids), mol_feat_all.size(1))
    mol_count = torch.zeros(len(unique_ids), dtype=torch.long)
    for i, mid in enumerate(all_mol_ids):
        idx = id_to_unique[mid]
        mol_feat_unique[idx] += mol_feat_all[i]
        mol_count[idx] += 1
    mol_feat_unique = mol_feat_unique / mol_count.unsqueeze(1).float()
    mol_feat_unique = nn.functional.normalize(mol_feat_unique, dim=-1)

    recall = model.compute_retrieval_metrics(
        ms_feat_all, mol_feat_unique,
        ms_labels=ms_labels,
        mol_labels=mol_labels_unique,
        ks=[1, 5, 10],
        device='cpu',
    )
    return recall


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Output dir with timestamp ──
    timestamp = time.strftime('%m%d_%H%M')
    run_name = args.run_name or f'clip_d{args.proj_dim}'
    output_dir = os.path.join(args.output_dir, f'retrieval_{run_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Tee logging
    log_path = os.path.join(output_dir, 'training.log')
    log_file = open(log_path, 'w', buffering=1)
    original_stdout = sys.stdout
    class Tee:
        def write(self, text): original_stdout.write(text); log_file.write(text)
        def flush(self): original_stdout.flush(); log_file.flush()
    sys.stdout = Tee()
    print(f'Output dir: {output_dir}')
    print(f'Log: {log_path}')

    # Save args
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f'Args: {vars(args)}')

    cache_dir = args.cache_dir or output_dir

    # ── Datasets ──
    print('\n=== Loading datasets ===')
    train_dataset = MSMolRetrievalDataset(
        hdf5_path=args.hdf5_path, split='train',
        cache_dir=cache_dir,
        force_recompute=args.force_molformer_recompute,
    )
    val_dataset = MSMolRetrievalDataset(
        hdf5_path=args.hdf5_path, split='val',
        cache_dir=cache_dir,
    )
    test_dataset = MSMolRetrievalDataset(
        hdf5_path=args.hdf5_path, split='test',
        cache_dir=cache_dir,
    )

    # Subset for quick iteration
    train_dataset_full = train_dataset  # keep full dataset ref for curriculum learning
    if args.subset is not None:
        n_train = min(args.subset, len(train_dataset))
        # When using subset with Subset wrapper, cluster_ids need special handling.
        # For simplicity with subset, just use standard DataLoader.
        train_dataset = torch.utils.data.Subset(train_dataset, range(n_train))
        print(f'[Subset] Using {n_train} training samples (standard DataLoader)')
        effective_hard_ratio = 0.0
    else:
        effective_hard_ratio = args.hard_ratio

    # Determine effective curriculum
    use_curriculum = (args.hard_ratio_phase2 is not None and args.hard_ratio_phase2 > 0
                      and args.switch_epoch < args.max_epochs)
    if use_curriculum:
        print(f'[Curriculum] Phase 1 (ep 1-{args.switch_epoch}): hard_ratio=0.0 (coarse alignment)')
        print(f'[Curriculum] Phase 2 (ep {args.switch_epoch+1}-{args.max_epochs}): '
              f'hard_ratio={args.hard_ratio_phase2} (fine-grained)')
        effective_hard_ratio = 0.0
    else:
        effective_hard_ratio = args.hard_ratio

    # Build train loader
    if effective_hard_ratio > 0.0 and hasattr(train_dataset_full, 'cluster_ids'):
        from ms2mol_retrieval.sampler import HardNegativeBatchSampler
        sampler = HardNegativeBatchSampler(
            cluster_ids=train_dataset_full.cluster_ids,
            batch_size=args.batch_size,
            hard_ratio=effective_hard_ratio,
        )
        train_loader = DataLoader(
            train_dataset_full, batch_sampler=sampler,
            num_workers=args.num_workers,
        )
        print(f'[Train] HardNegativeBatchSampler: '
              f'hard_ratio={effective_hard_ratio:.2f} '
              f'({sampler.hard_size} hard + {sampler.random_size} random per batch)')
    else:
        train_loader = DataLoader(
            train_dataset if args.subset else train_dataset_full,
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, drop_last=True,
        )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    print(f'Train: {len(train_dataset)} samples ({len(train_loader)} batches)')
    print(f'Val:   {len(val_dataset)} samples ({len(val_loader)} batches)')
    print(f'Test:  {len(test_dataset)} samples')

    # ── Model ──
    print('\n=== Building model ===')
    model = MSMolCLIP(
        ms_dim=args.ms_dim,
        mol_dim=args.mol_dim,
        proj_dim=args.proj_dim,
        proj_hidden=args.proj_hidden,
        proj_depth=args.proj_depth,
    ).to(device)

    # Resume from checkpoint
    if args.resume:
        print(f'Resuming from {args.resume}...')
        state = torch.load(args.resume, map_location=device)
        missing, unexpected = model.load_state_dict(
            state['model_state_dict'], strict=False,
        )
        if missing:
            print(f'  Missing keys: {len(missing)}')
        if unexpected:
            print(f'  Unexpected keys: {len(unexpected)}')

    # ── Optimizer & scheduler ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.max_epochs
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f'\n=== Training ===')
    print(f'Total steps: {total_steps} ({args.max_epochs} epochs × {len(train_loader)} batches)')
    print(f'Warmup: {warmup_steps} steps')
    print(f'Batch size: {args.batch_size}')
    print(f'LR: {args.lr} | Weight decay: {args.weight_decay}')

    # ── Wandb ──
    wandb_run = None
    if not args.no_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f'retrieval_{run_name}_{timestamp}',
            config=vars(args),
        )

    # ── Training loop ──
    best_val_loss = float('inf')
    best_recall = 0.0
    step_global = 0

    for epoch in range(args.max_epochs):
        epoch_start = time.time()

        # ── Curriculum: switch from phase 1 to phase 2 ──
        if use_curriculum and epoch == args.switch_epoch:
            print(f'\n=== Curriculum switch at epoch {epoch+1}: '
                  f'hard_ratio 0.0 → {args.hard_ratio_phase2} ===')
            from ms2mol_retrieval.sampler import HardNegativeBatchSampler
            sampler = HardNegativeBatchSampler(
                cluster_ids=train_dataset_full.cluster_ids,
                batch_size=args.batch_size,
                hard_ratio=args.hard_ratio_phase2,
            )
            train_loader = DataLoader(
                train_dataset_full, batch_sampler=sampler,
                num_workers=args.num_workers,
            )
            print(f'[Phase 2] HardNegativeBatchSampler: '
                  f'{sampler.hard_size} hard + {sampler.random_size} random per batch\n')
            # Recompute total_steps for remaining epochs
            remaining = args.max_epochs - epoch
            total_steps = step_global + remaining * len(train_loader)

        model.train()
        total_loss = 0
        total_acc = 0
        total_recall5 = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            ms_emb = batch['ms_emb'].to(device)
            mol_emb = batch['mol_emb'].to(device)

            optimizer.zero_grad()
            ms_feat, mol_feat = model(ms_emb, mol_emb)
            loss_dict = model.compute_loss(ms_feat, mol_feat)
            loss = loss_dict['loss']
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            scheduler.step()
            step_global += 1

            total_loss += loss.item()
            total_acc += loss_dict['acc']
            total_recall5 += loss_dict['recall_ms@5']
            n_batches += 1

            # Log to wandb every 50 steps
            if (batch_idx + 1) % 50 == 0 and wandb_run:
                wandb_run.log({
                    'train/loss': loss.item(),
                    'train/acc': loss_dict['acc'],
                    'train/recall_ms@5': loss_dict['recall_ms@5'],
                    'train/logit_scale': loss_dict['logit_scale'],
                    'train/lr': scheduler.get_last_lr()[0],
                    'step': step_global,
                })

        # Epoch metrics
        train_loss = total_loss / max(n_batches, 1)
        train_acc = total_acc / max(n_batches, 1)
        train_recall5 = total_recall5 / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        # Validation loss
        model.eval()
        val_total_loss = 0
        val_total_acc = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                ms_emb = batch['ms_emb'].to(device)
                mol_emb = batch['mol_emb'].to(device)
                ms_feat, mol_feat = model(ms_emb, mol_emb)
                loss_dict = model.compute_loss(ms_feat, mol_feat)
                val_total_loss += loss_dict['loss'].item()
                val_total_acc += loss_dict['acc']
                val_n += 1
        val_loss = val_total_loss / max(val_n, 1)
        val_acc = val_total_acc / max(val_n, 1)

        lr_now = scheduler.get_last_lr()[0]
        tau = 1.0 / loss_dict.get('logit_scale', 14.29) if n_batches > 0 else 0.07

        print(
            f'Epoch {epoch+1}/{args.max_epochs} | '
            f'train_loss={train_loss:.4f} train_acc={train_acc:.4f} rec@5={train_recall5:.4f} | '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} | '
            f'τ={tau:.4f} lr={lr_now:.2e} | {epoch_time:.0f}s'
        )

        # ── FAISS full-database evaluation ──
        if not args.no_faiss_eval and (epoch + 1) % args.eval_every == 0:
            print(f'  --- FAISS full-database evaluation at epoch {epoch+1} ---')
            recall = build_val_index(model, val_dataset, device)
            recall_str = ' | '.join([f'{k}={v:.4f}' for k, v in recall.items()])
            print(f'  Recall: {recall_str}')
            if wandb_run:
                wandb_log = {f'val/{k}': v for k, v in recall.items()}
                wandb_log['epoch'] = epoch + 1
                wandb_run.log(wandb_log)

            # Track best recall
            current_recall = recall.get('recall@1', 0.0)
            if current_recall > best_recall:
                best_recall = current_recall
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'recall': recall,
                }, os.path.join(output_dir, 'best.ckpt'))
                print(f'  ✓ New best recall@1={current_recall:.4f}')

        # Save checkpoint every epoch
        is_best_val = val_loss < best_val_loss
        if is_best_val:
            best_val_loss = val_loss

        ckpt = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        torch.save(ckpt, os.path.join(output_dir, 'last.ckpt'))
        if is_best_val:
            torch.save(ckpt, os.path.join(output_dir, 'best_val.ckpt'))
            print(f'  ✓ New best val_loss={val_loss:.4f}')

        if wandb_run:
            wandb_run.log({
                'epoch/train_loss': train_loss,
                'epoch/train_acc': train_acc,
                'epoch/train_recall@5': train_recall5,
                'epoch/val_loss': val_loss,
                'epoch/val_acc': val_acc,
                'epoch/lr': lr_now,
                'epoch': epoch + 1,
            })

    # ── Final test evaluation ──
    print('\n=== Final test evaluation ===')
    best_path = os.path.join(output_dir, 'best.ckpt')
    if not os.path.exists(best_path):
        best_path = os.path.join(output_dir, 'last.ckpt')
        print(f'  (using last.ckpt — best.ckpt not saved without --no_faiss_eval)')
    model.load_state_dict(torch.load(best_path, map_location=device)['model_state_dict'])
    test_recall = build_val_index(model, test_dataset, device)
    for k, v in test_recall.items():
        print(f'  Test {k}: {v:.4f}')
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_recall, f, indent=2)

    print('\n=== Done! ===')
    print(f'Best val_loss: {best_val_loss:.4f}')
    print(f'Best recall@1: {best_recall:.4f}')
    print(f'Output: {output_dir}')

    if wandb_run:
        wandb_run.finish()


if __name__ == '__main__':
    main()
