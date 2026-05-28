"""N-sample overfitting test.

DEPRECATED: Use `python -m ms2mol_encdec.train --n N` instead.

Trains on a small subset (default: 100 samples) to verify the model
can memorize N different MS→SMILES mappings simultaneously.

If it can reach high exact match rate on N samples, the architecture
has sufficient capacity and the training pipeline is correct.
Low Tanimoto on the full dataset is then a generalization problem,
not a capacity or code problem.

Usage:
    # Phase 1 test with 100 samples
    python -m ms2mol_encdec.test_overfit_n --n 100 --phase 1

    # Phase 2 test with 100 samples
    python -m ms2mol_encdec.test_overfit_n --n 100 --phase 2

    # Both phases
    python -m ms2mol_encdec.test_overfit_n --n 100 --phase both
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
os.environ['TRANSFORMERS_CACHE'] = '/tmp/t5cache'

from ms2mol_encdec.model import MSToSMILES_T5, info_display, MODEL_REGISTRY
from ms2mol_encdec.dataset import T5SmilesCollator, smi_to_maccs, NSubsetDataset
from ms2mol_shared.lora import inject_lora_t5, count_lora_params


# ── N-sample dataset ─────────────────────────────────────────────

class Tee:
    """Tee stdout to both terminal and a log file."""
    def __init__(self, log_path: str):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.file = open(log_path, 'w', buffering=1)
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


# ── Evaluation ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, dataset, tokenizer, device, num_beams=5, max_length=200):
    """Evaluate exact match and Tanimoto on the entire dataset."""
    model.eval()
    n = len(dataset)

    all_embs = torch.stack([dataset[i]['ms_emb'] for i in range(n)]).to(device)
    refs = [dataset[i]['smiles'] for i in range(n)]

    batch_size = 64
    all_generated = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        gen = model.generate(
            ms_emb=all_embs[start:end],
            num_beams=num_beams,
            max_length=max_length,
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
            tan = 0.0

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


# ── Main test ────────────────────────────────────────────────────

def test_overfit_n(args):
    # ── Setup training log ──
    if args.output_dir:
        log_path = os.path.join(args.output_dir, 'training.log')
        sys.stdout = Tee(log_path)
        print(f'Log: {log_path}')

    device = torch.device(args.device)
    print(f'Device: {device}')
    print(f'Model: {info_display(args.model_name)} (--model_name={args.model_name})')
    print(f'Samples: {args.n}')
    print(f'K tokens: {args.k_tokens}')
    print()

    # ── Dataset ──
    dataset = NSubsetDataset(n=args.n, seed=args.seed)
    collator = T5SmilesCollator(tokenizer=None, max_length=512)  # dummy, we'll tokenize manually

    # We need a real tokenizer — load it from the model
    print()

    # ── Model ──
    model = MSToSMILES_T5(
        k_tokens=args.k_tokens, model_name=args.model_name,
        projector_type=args.projector_type,
        projector_depth=args.projector_depth,
        projector_trunk_dim=args.projector_trunk_dim,
        projector_head_rank=args.projector_head_rank,
        ms_decoder_adapter=args.ms_decoder_adapter,
        maccs_loss_weight=args.maccs_loss_weight,
        ce_loss_weight=args.ce_loss_weight,
    ).to(device)
    tokenizer = model.tokenizer
    collator.tokenizer = tokenizer  # patch collator

    # Phase setup
    if args.phase in ('1', 'both_1'):
        for p in model.t5.parameters():
            p.requires_grad = False
        print('[Phase 1] Backbone: FROZEN, Projector: TRAIN')
    elif args.phase in ('2', 'both_2'):
        for p in model.t5.parameters():
            p.requires_grad = False
        inject_lora_t5(model.t5, rank=args.lora_rank, alpha=args.lora_alpha)
        print(f'[Phase 2] Backbone: FROZEN + LoRA (r={args.lora_rank}), Projector: TRAIN')

    # ── Load projector checkpoint (for Phase 2 with trained projector) ──
    if args.projector_ckpt:
        print(f'Loading projector from {args.projector_ckpt}')
        proj_state = torch.load(args.projector_ckpt, map_location=device)
        model.projector.load_state_dict(proj_state)
        print(f'  Loaded projector state dict ({len(proj_state)} keys)')
        if args.freeze_projector:
            for p in model.projector.parameters():
                p.requires_grad = False
            print('  ✅ Projector FROZEN — only LoRA will train')

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.2f}%)')
    print()

    # ── DataLoader ──
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=0,
    )
    print(f'Batches per epoch: {len(loader)} (batch_size={args.batch_size})')

    # ── Optimizer ──
    param_groups = []
    proj_trainable = any(p.requires_grad for p in model.projector.parameters())
    if proj_trainable:
        proj_lr = args.lr
        if args.phase in ('2', 'both_2') and hasattr(args, 'projector_lr_scale'):
            if args.projector_lr_scale > 0:
                proj_lr = args.lr * args.projector_lr_scale
                print(f'[Optimizer] Phase 2: projector LR = {proj_lr:.2e} '
                      f'(scale={args.projector_lr_scale})')
            else:
                for p in model.projector.parameters():
                    p.requires_grad = False
                proj_trainable = False
                print('[Optimizer] Phase 2: projector FROZEN (projector_lr_scale=0)')
        if proj_trainable:
            proj_params = list(model.projector.parameters())
            if hasattr(model, 'maccs_head') and model.maccs_head is not None:
                proj_params += list(model.maccs_head.parameters())
            param_groups.append({'params': proj_params, 'lr': proj_lr})
    total_steps = len(loader) * args.epochs
    if args.phase in ('2',):
        lora_params = [p for n, p in model.t5.named_parameters()
                       if 'lora_A' in n or 'lora_B' in n]
        if lora_params:
            param_groups.append({'params': lora_params, 'lr': args.lr_lora})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0)

    from transformers import get_cosine_schedule_with_warmup
    warmup_steps = min(args.warmup_steps, total_steps // 10)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ── Training ──
    print(f'\nTraining for {args.epochs} epochs ({total_steps} steps)...')
    print(f'  lr_projector={args.lr:.2e}, lr_lora={args.lr_lora:.2e}')
    print(f'  warmup={warmup_steps} steps')
    print(f'  Evaluate every {args.eval_every} epochs, early stop at exact_match >= {args.target_exact}')
    print()

    best_exact = 0.0
    best_metrics = None
    target_epoch = None
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_maccs_correct = 0
        total_maccs_bits = 0
        n_batches = 0

        for batch in loader:
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
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            # Track MACCS metrics
            if outputs.get('maccs_logits') is not None and maccs is not None:
                bc = outputs['maccs_loss']
                if bc is not None:
                    preds = (outputs['maccs_logits'] > 0).float()
                    total_maccs_correct += (preds == maccs).sum().item()
                    total_maccs_bits += maccs.numel()

        train_loss = total_loss / max(n_batches, 1)

        # Evaluation
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics, eval_results = evaluate(model, dataset, tokenizer, device, num_beams=args.num_beams)
            elapsed = time.time() - start_time

            exact = metrics['exact_match_rate']
            tanimoto = metrics['tanimoto_mean']
            valid = metrics['valid_rate']

            mark = '✓' if exact >= args.target_exact else ' '
            log = (f'  [{mark}] Epoch {epoch:3d}/{args.epochs} | '
                   f'loss={train_loss:.4f} | '
                   f'exact={exact*100:.2f}% | '
                   f'valid={valid*100:.1f}% | '
                   f'tanimoto={tanimoto:.4f}')
            if total_maccs_bits > 0:
                maccs_acc = total_maccs_correct / total_maccs_bits * 100
                log += f' | MACCS={maccs_acc:.1f}%'
            log += f' | {elapsed:.0f}s'
            print(log)

            if exact >= args.target_exact:
                print(f'  >>> TARGET EXACT MATCH ({args.target_exact*100:.0f}%) REACHED at epoch {epoch}! <<<')
                if target_epoch is None:
                    target_epoch = epoch

            if exact > best_exact:
                best_exact = exact
                best_metrics = metrics

            if args.stop_on_target and exact >= args.target_exact:
                print(f'  Early stop: exact match >= {args.target_exact*100:.0f}%')
                break

            # ── Save checkpoint periodically (every eval epoch) ──
            if args.save_projector and args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                ckpt = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'train_loss': train_loss,
                    'best_exact': best_exact,
                    'phase': args.phase,
                    'model_name': args.model_name,
                    'k_tokens': args.k_tokens,
                    'projector_type': args.projector_type,
                    'projector_depth': args.projector_depth,
                    'projector_trunk_dim': args.projector_trunk_dim,
                    'projector_head_rank': args.projector_head_rank,
                    'target_epoch': target_epoch,
                    'trainable': trainable,
                    'total_params': total,
                    'n': args.n,
                    'seed': args.seed,
                    'projector_dropout': args.projector_dropout,
                    'ms_decoder_adapter': args.ms_decoder_adapter,
                    'maccs_loss_weight': args.maccs_loss_weight,
                    'ce_loss_weight': args.ce_loss_weight,
                }
                ckpt_path = os.path.join(args.output_dir, 'last.ckpt')
                torch.save(ckpt, ckpt_path)

                # Also save projector standalone for convenient loading
                proj_path = os.path.join(args.output_dir, 'projector.pt')
                torch.save(model.projector.state_dict(), proj_path)
                print(f'  Saved checkpoint @ epoch {epoch}: {ckpt_path}')
                print(f'  Saved projector weights @ epoch {epoch}: {proj_path}')

    # ── Summary ──
    total_time = time.time() - start_time
    print()
    print('=' * 60)
    print(f'{args.n}-SAMPLE OVERFITTING TEST RESULTS ({args.phase_name})')
    print('=' * 60)
    print(f'  Samples:              {args.n}')
    print(f'  Trainable params:     {trainable:,} / {total:,}')
    print(f'  Final train loss:     {train_loss:.4f}')
    print(f'  Best exact match:     {best_exact*100:.2f}%')
    print(f'  Best valid SMILES:    {best_metrics["valid_rate"]*100:.1f}%' if best_metrics else '  N/A')
    print(f'  Best Tanimoto mean:   {best_metrics["tanimoto_mean"]:.4f}' if best_metrics else '  N/A')
    print(f'  Target epoch:         {target_epoch}')
    print(f'  Total time:           {total_time:.0f}s ({total_time/60:.1f}min)')
    print(f'  Target met?           {"YES ✓" if target_epoch is not None else "NO ✗"}')
    print()

    result = {
        'n': args.n,
        'best_exact': best_exact,
        'best_valid': best_metrics['valid_rate'] if best_metrics else 0,
        'best_tanimoto': best_metrics['tanimoto_mean'] if best_metrics else 0,
        'target_epoch': target_epoch,
        'passed': target_epoch is not None,
        'phase': args.phase_name,
        'k_tokens': args.k_tokens,
        'projector_type': args.projector_type,
        'projector_depth': args.projector_depth,
        'projector_head_rank': args.projector_head_rank,
        'epochs_run': epoch,
        'train_loss': train_loss,
    }
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        tag = f'n{args.n}_ph{args.phase}_{args.projector_type}'
        if args.projector_type == 'mlp':
            tag += f'_d{args.projector_depth}_k{args.k_tokens}'
        else:
            tag += f'_k{args.k_tokens}_r{args.projector_head_rank}'
        out_path = os.path.join(args.output_dir, f'{tag}_summary.json')
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f'  Saved summary: {out_path}')
    return result


def main():
    parser = argparse.ArgumentParser(description='N-sample overfitting test')
    parser.add_argument('--model_name', default='t5-small',
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--phase', default='both',
                        choices=['1', '2', 'both'])
    parser.add_argument('--n', type=int, default=100,
                        help='Number of training samples')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sample selection')
    parser.add_argument('--k_tokens', type=int, default=16)
    parser.add_argument('--projector_type', default='mlp',
                        choices=['mlp', 'k_heads'],
                        help='Projector: mlp (legacy) or k_heads (D1)')
    parser.add_argument('--projector_depth', type=int, default=2,
                        help='Depth of projector MLP when projector_type=mlp')
    parser.add_argument('--projector_trunk_dim', type=int, default=512)
    parser.add_argument('--projector_head_rank', type=int, default=64)
    parser.add_argument('--projector_dropout', type=float, default=0.0,
                        help='Dropout rate for projector layers')
    parser.add_argument('--ms_decoder_adapter', action='store_true',
                        help='Add MS decoder adapter: project MS to d_model and append to encoder output')

    # ── Substructure supervision (MACCS keys) ──
    parser.add_argument('--maccs_loss_weight', type=float, default=0.0,
                        help='Weight for MACCS BCE loss (default: 0.0 = disabled)')
    parser.add_argument('--ce_loss_weight', type=float, default=1.0,
                        help='Weight for T5 CE loss (default: 1.0)')

    parser.add_argument('--epochs', type=int, default=300,
                        help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_lora', type=float, default=3e-4)
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=float, default=16.0)
    parser.add_argument('--projector_lr_scale', type=float, default=0.1,
                        help='Phase 2: scale projector LR = lr * projector_lr_scale '
                             '(default: 0.1). Set 0.0 to freeze projector.')
    parser.add_argument('--warmup_steps', type=int, default=50)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Evaluate every N epochs')
    parser.add_argument('--num_beams', type=int, default=5,
                        help='Beam size for generation eval')
    parser.add_argument('--target_exact', type=float, default=0.80,
                        help='Target exact match rate (e.g., 0.80 = 80%)')
    parser.add_argument('--stop_on_target', action='store_true',
                        help='Stop training when target_exact is reached')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save summary JSON')
    parser.add_argument('--save_projector', action='store_true',
                        help='Save projector weights to output_dir/projector.pt after Phase 1')
    parser.add_argument('--projector_ckpt', type=str, default=None,
                        help='Load projector weights from this checkpoint file')
    parser.add_argument('--freeze_projector', action='store_true',
                        help='Freeze projector after loading (Phase 2 only-LoRA mode)')
    args = parser.parse_args()

    results = {}
    phases_to_run = []

    if args.phase == 'both':
        phases_to_run = [('1', 'Phase 1 (projector only)'),
                         ('2', 'Phase 2 (projector + LoRA)')]
    else:
        phases_to_run = [(args.phase, f'Phase {args.phase}')]

    for phase_num, phase_name in phases_to_run:
        args.phase = phase_num
        args.phase_name = phase_name

        print('\n' + '#' * 60)
        print(f'# {phase_name}')
        print('#' * 60)

        result = test_overfit_n(args)
        results[phase_name] = result

        if result['passed']:
            print(f'  ✅ {phase_name}: PASSED ({result["best_exact"]*100:.1f}% exact, '
                  f'Tanimoto={result["best_tanimoto"]:.4f})')
        else:
            print(f'  ❌ {phase_name}: FAILED (best exact={result["best_exact"]*100:.1f}%)')

    print('\n' + '=' * 60)
    print(f'FINAL VERDICT — {args.n}-sample overfitting test')
    print('=' * 60)
    all_passed = all(r['passed'] for r in results.values())
    if all_passed:
        print(f'  ✅ PASSED — model can memorize {args.n} MS→SMILES mappings')
        print(f'     Best exact: {max(r["best_exact"] for r in results.values())*100:.1f}%')
        print(f'     Best Tanimoto: {max(r["best_tanimoto"] for r in results.values()):.4f}')
        print(f'     The architecture has sufficient capacity.')
        print(f'     Low Tanimoto (~0.12) on full 313k dataset is a generalization problem.')
    else:
        failed = [k for k, v in results.items() if not v['passed']]
        print(f'  ⚠️  PARTIAL FAILURE: {len(failed)} test(s) below target ({failed})')
        print(f'     This may indicate a capacity or optimization bottleneck.')


if __name__ == '__main__':
    main()
