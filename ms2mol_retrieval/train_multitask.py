"""Multi-task training: InfoNCE (512-d) + MACCS + molecular weight.

Architecture (CLIP-style 512-d shared space + 2 auxiliary heads):
    ms_emb (1024) → MS Projector → L2Norm → 512-d ──┐
    mol_emb (768) → MoL Projector → L2Norm → 512-d ──┤── InfoNCE
    ms_emb (1024) → MACCS Proj (256) → 166-d ────────┤── BCE
    ms_emb (1024) → MW Proj (64) → 1-d ──────────────┘── Huber

Usage:
    # Smoke test (40k subset, 5 epochs)
    python -m ms2mol_retrieval.train_multitask \\
        --subset 40000 --max_epochs 5 --batch_size 256

    # Full training (50 epochs, bs8192)
    python -m ms2mol_retrieval.train_multitask \\
        --batch_size 8192 --max_epochs 50 --proj_depth 3

    # Shared trunk mode (40k subset, 5 epochs)
    python -m ms2mol_retrieval.train_multitask \\
        --use_shared_trunk --subset 40000 --max_epochs 5 --batch_size 256

    # Shared trunk full training
    python -m ms2mol_retrieval.train_multitask \\
        --use_shared_trunk --batch_size 8192 --max_epochs 50 --proj_depth 3
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from ms2mol_retrieval.model import MSMolCLIPMultiTask, MSMolCLIPSharedTrunk
from ms2mol_retrieval.dataset import MultiTaskRetrievalDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-task training (512-d CLIP style)')

    # Data
    parser.add_argument('--hdf5_path', default='/root/datasets/pairs_with_embs.hdf5')
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--cache_dir', type=str, default='/root/DreaMS/ms2mol_retrieval/shared_cache')
    parser.add_argument('--num_workers', type=int, default=0)

    # Architecture (CLIP-style projector)
    parser.add_argument('--proj_dim', type=int, default=512)
    parser.add_argument('--proj_hidden', type=int, default=1024)
    parser.add_argument('--proj_depth', type=int, default=2, choices=[2, 3])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_shared_trunk', action='store_true',
                        help='Use MSMolCLIPSharedTrunk (shared MLP before 3 heads)')
    parser.add_argument('--trunk_dim', type=int, default=512,
                        help='Shared trunk output dimension (only for --use_shared_trunk)')

    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_logit', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--grad_clip', type=float, default=5.0)

    # Loss weights
    parser.add_argument('--w_cross', type=float, default=1.0,
                        help='InfoNCE loss weight')
    parser.add_argument('--w_maccs', type=float, default=2.0,
                        help='MACCS BCE loss weight')
    parser.add_argument('--w_mw', type=float, default=5.0,
                        help='Molecular weight Huber loss weight')

    # Evaluation
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--no_faiss_eval', action='store_true')

    # Output
    parser.add_argument('--output_dir', default='/root/DreaMS/ms2mol_retrieval/outputs')
    parser.add_argument('--run_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--no_wandb', action='store_true', default=True)

    return parser.parse_args()


def compute_multitask_loss(model, outputs, targets, weights):
    """Compute 3-task joint loss.

    Args:
        model: MSMolCLIPMultiTask (for compute_loss + logit_scale).
        outputs: dict from model.forward(ms_emb, mol_emb).
        targets: dict with 'mol_feat', 'maccs', 'mol_weight'.
        weights: dict with 'w_cross', 'w_maccs', 'w_mw'.
    Returns:
        dict with 'total_loss', 'loss_cross', 'loss_maccs', 'loss_mw', 'acc_cross'.
    """
    # Task 1: InfoNCE (uses model.compute_loss which is symmetric CE)
    loss_dict = model.compute_loss(outputs['ms_feat'], outputs['mol_feat'])
    loss_cross = loss_dict['loss']

    # Task 2: MACCS BCE
    loss_maccs = F.binary_cross_entropy_with_logits(
        outputs['maccs_logits'], targets['maccs'].float(),
    )

    # Task 3: Molecular weight Huber
    loss_mw = F.smooth_l1_loss(
        outputs['mol_weight'].squeeze(-1), targets['mol_weight'].squeeze(-1), beta=1.0,
    )

    # Combined
    total_loss = (
        weights['w_cross'] * loss_cross
        + weights['w_maccs'] * loss_maccs
        + weights['w_mw'] * loss_mw
    )

    return {
        'total_loss': total_loss,
        'loss_cross': loss_cross.detach(),
        'loss_maccs': loss_maccs.detach(),
        'loss_mw': loss_mw.detach(),
        'acc_cross': loss_dict['acc'],
        'logit_scale': loss_dict['logit_scale'],
    }


def evaluate(model, val_loader, weights, device, prefix='val'):
    """Evaluate on validation set."""
    model.eval()
    totals = {'loss': 0.0, 'cross': 0.0, 'maccs': 0.0, 'mw': 0.0,
              'acc_cross': 0.0, 'acc_maccs': 0.0, 'mw_mae_z': 0.0}
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            ms_emb = batch['ms_emb'].to(device)
            mol_emb = batch['mol_emb'].to(device)
            maccs = batch['maccs'].to(device)
            mw = batch['mol_weight'].to(device)

            outputs = model(ms_emb, mol_emb)
            losses = compute_multitask_loss(model, outputs,
                {'mol_feat': mol_emb, 'maccs': maccs, 'mol_weight': mw}, weights)

            totals['loss'] += losses['total_loss'].item()
            totals['cross'] += losses['loss_cross'].item()
            totals['maccs'] += losses['loss_maccs'].item()
            totals['mw'] += losses['loss_mw'].item()
            totals['acc_cross'] += losses['acc_cross']
            totals['acc_maccs'] += ((outputs['maccs_logits'] > 0).float() == maccs).float().mean().item()
            totals['mw_mae_z'] += F.l1_loss(outputs['mol_weight'].squeeze(), mw.squeeze()).item()
            n += 1

    return {f'{prefix}/{k}': v / n for k, v in totals.items()}


def evaluate_retrieval(model, dataset, device, batch_size=512):
    """Full-database MS→Mol retrieval via FAISS (512-d space)."""
    import faiss
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_ms, all_mol, all_ids = [], [], []

    with torch.no_grad():
        for batch in loader:
            ms_emb = batch['ms_emb'].to(device)
            mol_emb = batch['mol_emb'].to(device)
            all_ms.append(model.encode_ms(ms_emb).cpu())
            all_mol.append(model.encode_mol(mol_emb).cpu())
            all_ids.extend(batch['mol_id'].numpy().tolist())

    ms_feat_all = torch.cat(all_ms, dim=0)
    mol_feat_all = torch.cat(all_mol, dim=0)

    # Deduplicate molecules
    unique_ids = sorted(set(all_ids))
    id_map = {uid: i for i, uid in enumerate(unique_ids)}
    feat_unique = torch.zeros(len(unique_ids), ms_feat_all.size(1))
    count = torch.zeros(len(unique_ids), dtype=torch.long)
    for i, mid in enumerate(all_ids):
        idx = id_map[mid]
        feat_unique[idx] += mol_feat_all[i]
        count[idx] += 1
    feat_unique = F.normalize(feat_unique / count.unsqueeze(1).float(), dim=-1)

    index = faiss.IndexFlatIP(ms_feat_all.size(1))
    index.add(feat_unique.numpy().astype(np.float32))
    _, indices = index.search(ms_feat_all.numpy().astype(np.float32), 10)

    recall = {}
    for k in [1, 5, 10]:
        hits = sum(1 for i in range(len(ms_feat_all))
                   if all_ids[i] in [unique_ids[idx] for idx in indices[i, :k]])
        recall[f'recall@{k}'] = hits / len(ms_feat_all)
    return recall


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Output ──
    timestamp = time.strftime('%m%d_%H%M')
    run_name = args.run_name or f'{"st_" if args.use_shared_trunk else "mt_"}{args.proj_dim}d_b{args.batch_size}'
    output_dir = os.path.join(args.output_dir, f'{run_name}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, 'training.log')
    log_file = open(log_path, 'w', buffering=1)
    orig_stdout = sys.stdout

    class Tee:
        def write(self, text): orig_stdout.write(text); log_file.write(text)
        def flush(self): orig_stdout.flush(); log_file.flush()
    sys.stdout = Tee()
    print(f'Output: {output_dir}')
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── Data ──
    print('\n=== Loading data ===')
    train_dataset = MultiTaskRetrievalDataset(hdf5_path=args.hdf5_path, split='train', cache_dir=args.cache_dir)
    val_dataset = MultiTaskRetrievalDataset(hdf5_path=args.hdf5_path, split='val', cache_dir=args.cache_dir)
    test_dataset = MultiTaskRetrievalDataset(hdf5_path=args.hdf5_path, split='test', cache_dir=args.cache_dir)

    if args.subset:
        train_dataset = Subset(train_dataset, range(min(args.subset, len(train_dataset))))
        print(f'[Subset] {len(train_dataset)} samples')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=min(args.batch_size, 1024), shuffle=False,
                            num_workers=args.num_workers)
    print(f'Train: {len(train_dataset)} ({len(train_loader)} batches)')
    print(f'Val:   {len(val_dataset)} ({len(val_loader)} batches)')
    print(f'Test:  {len(test_dataset)}')

    # ── Model ──
    print('\n=== Building model ===')
    if args.use_shared_trunk:
        ModelClass = MSMolCLIPSharedTrunk
        print(f'[Model] Shared trunk: ms_emb → {args.trunk_dim}d trunk → 3 heads')
    else:
        ModelClass = MSMolCLIPMultiTask
        print(f'[Model] Separate heads from raw 1024-d ms_emb')
    model = ModelClass(
        ms_dim=1024, mol_dim=768,
        proj_dim=args.proj_dim, proj_hidden=args.proj_hidden,
        proj_depth=args.proj_depth, dropout=args.dropout,
        **{'trunk_dim': args.trunk_dim} if args.use_shared_trunk else {},
    ).to(device)

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state['model_state_dict'])
        print(f'Resumed from {args.resume} (epoch {state.get("epoch", "?")})')

    # ── Optimizer (纯 CLIP 风格：所有参数统一 lr，不含单独 logit_scale 分组) ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.max_epochs
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    loss_weights = {'w_cross': args.w_cross, 'w_maccs': args.w_maccs, 'w_mw': args.w_mw}
    print(f'\n=== Training ({args.max_epochs} epochs, bs={args.batch_size}) ===')
    print(f'Loss weights: cross={args.w_cross}, maccs={args.w_maccs}, mw={args.w_mw}')

    best_val_loss = float('inf')
    best_recall_r1 = 0.0
    global_step = 0

    for epoch in range(args.max_epochs):
        epoch_start = time.time()
        model.train()
        train_losses = []

        for batch in train_loader:
            ms_emb = batch['ms_emb'].to(device)
            mol_emb = batch['mol_emb'].to(device)
            maccs = batch['maccs'].to(device)
            mw = batch['mol_weight'].to(device)

            optimizer.zero_grad()
            outputs = model(ms_emb, mol_emb)
            targets = {'mol_feat': mol_emb, 'maccs': maccs, 'mol_weight': mw}
            losses = compute_multitask_loss(model, outputs, targets, loss_weights)
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            train_losses.append(losses['total_loss'].item())

            if global_step % 100 == 0:
                tau = 1.0 / losses['logit_scale']
                print(f'  Ep{epoch+1} step{global_step} | loss={losses["total_loss"]:.4f} '
                      f'cross={losses["loss_cross"]:.4f} maccs={losses["loss_maccs"]:.4f} '
                      f'mw={losses["loss_mw"]:.4f} τ={tau:.4f}', flush=True)

        # ── Epoch summary ──
        val_metrics = evaluate(model, val_loader, loss_weights, device)
        lr_now = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch+1}/{args.max_epochs} | '
              f'train_loss={np.mean(train_losses):.4f} '
              f'val_loss={val_metrics["val/loss"]:.4f} '
              f'acc_cross={val_metrics["val/acc_cross"]:.4f} '
              f'acc_maccs={val_metrics["val/acc_maccs"]:.4f} '
              f'mw_mae_z={val_metrics["val/mw_mae_z"]:.4f} | '
              f'lr={lr_now:.2e} | {time.time()-epoch_start:.0f}s')

        # ── Checkpoint ──
        is_best = val_metrics['val/loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val/loss']

        # ── FAISS eval ──
        if not args.no_faiss_eval and (epoch + 1) % args.eval_every == 0:
            recall = evaluate_retrieval(model, val_dataset, device)
            recall_str = ' | '.join(f'{k}={v:.4f}' for k, v in recall.items())
            print(f'  FAISS val: {recall_str}')
            if recall.get('recall@1', 0) > best_recall_r1:
                best_recall_r1 = recall['recall@1']

        ckpt = {'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'args': vars(args)}
        torch.save(ckpt, os.path.join(output_dir, 'last.ckpt'))
        if is_best:
            torch.save(ckpt, os.path.join(output_dir, 'best.ckpt'))
            print(f'  ✓ Best val_loss={best_val_loss:.4f}')

    # ── Final test ──
    print('\n=== Test evaluation ===')
    best_path = os.path.join(output_dir, 'best.ckpt')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)['model_state_dict'])
    test_recall = evaluate_retrieval(model, test_dataset, device)
    for k, v in test_recall.items():
        print(f'  Test {k}: {v:.4f}')
    with open(os.path.join(output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_recall, f, indent=2)

    print(f'\nBest val_loss={best_val_loss:.4f} | Best recall@1={best_recall_r1:.4f}')
    print(f'Output: {output_dir}')


if __name__ == '__main__':
    main()
