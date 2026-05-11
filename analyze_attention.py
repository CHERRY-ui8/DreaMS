#!/usr/bin/env python3
"""
Attention Analysis for Cond-Token DreaMS Model

Extracts attention weights from a trained cond-token model and analyzes
how fragment peaks attend to adduct and CE tokens.

Usage:
    python analyze_attention.py [--ckpt /root/DreaMS/dreams_cond/v1/last-v1.ckpt]
                                [--hdf5 /root/datasets/dreams_ready.hdf5]
                                [--output /root/DreaMS/attention_results]
"""
import sys, os, argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Add DreaMS to path ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dreams.api import PreTrainedModel, DreaMSModel
from dreams.utils.data import MSData, SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from dreams.definitions import SPECTRUM, ADDUCT, COLLISION_ENERGY, PRECURSOR_MZ
from torch.utils.data import DataLoader
from argparse import Namespace


def build_adduct_vocab(hdf5_path):
    """Rebuild adduct vocabulary identical to MaskedSpectraDataset's method."""
    import h5py
    with h5py.File(hdf5_path, 'r') as f:
        adducts = f['adduct'][:]
    strings = set()
    for a in adducts:
        s = a.decode('utf-8') if isinstance(a, bytes) else a
        strings.add(s)
    vocab = ['<PAD>', '<UNK>'] + sorted(strings)
    str_to_idx = {s: i for i, s in enumerate(vocab)}
    print(f"  Rebuilt adduct vocabulary: {len(vocab)} tokens ({len(vocab)-2} unique adducts)")
    return vocab, str_to_idx


def load_cond_model(ckpt_path, hdf5_path, device='cpu'):
    """Load cond-token checkpoint with correct args."""
    from dreams.api import PreTrainedModel
    import warnings; warnings.filterwarnings('ignore')

    # First load raw checkpoint to extract args
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hp = raw['hyper_parameters']
    args_dict = vars(hp['args']) if hasattr(hp['args'], '__dict__') else hp['args']

    # Override original hparam dropout values with what was actually used
    # (the checkpoint was saved mid-training, hparams may differ)
    print(f"  Model hparams: train_objective={args_dict.get('train_objective')}, "
          f"enable_cond_tokens={args_dict.get('enable_cond_tokens')}")

    # Build adduct vocabulary to get str→idx mapping
    vocab, str_to_idx = build_adduct_vocab(hdf5_path)

    # Set cond-token params in args
    args_dict['enable_cond_tokens'] = True
    args_dict['adduct_vocab_size'] = len(vocab)
    args_dict['ce_max'] = 200.0

    # Ensure transformer bias matches checkpoint
    if 'no_transformer_bias' not in args_dict:
        args_dict['no_transformer_bias'] = True

    # Load model
    model = PreTrainedModel._load_dreams_checkpoint(ckpt_path, map_location=device,
                                                     args=Namespace(**args_dict))
    model = model.eval().to(device)
    print(f"  Model loaded on {device}: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  enable_cond_tokens={model.enable_cond_tokens}, "
          f"adduct_vocab_size={model.adduct_vocab_size}")

    return model, str_to_idx


def get_attention_hooks(model, layers_idx=None):
    """Register forward hooks on attention layers. Returns (hooks, storage dict)."""
    if layers_idx is None:
        layers_idx = list(range(model.n_layers))

    # Storage: {layer_idx: {'attn': tensor}}
    storage = {i: {} for i in layers_idx}

    hooks = []
    for i in layers_idx:
        att_layer = model.transformer_encoder.atts[i]

        def make_hook(idx):
            def hook(module, inp, out):
                # out = (attn_output, att_weights)
                # att_weights shape: (B, n_heads, seq_len, seq_len)
                storage[idx]['attn'] = out[1].detach().cpu()
            return hook

        hooks.append(att_layer.register_forward_hook(make_hook(i)))

    return hooks, storage


def run_attention_on_spectra(model, msdata, indices, str_to_idx, device='cpu',
                             batch_size=1):
    """
    Run single-spectrum forward passes with attention hooks.
    Returns dict: {idx: {'attn_layer_i': (B, n_heads, S, S), 'adduct': str, 'ce': float}}
    """
    from torch.utils.data import Subset
    spec_preproc = SpectrumPreprocessor(
        dformat=DataFormatA(), n_highest_peaks=60
    )

    dataset = msdata.to_torch_dataset(spec_preproc)
    indices_sorted = sorted(set(indices))
    subset = Subset(dataset, indices_sorted)
    loader = DataLoader(subset, batch_size=1, shuffle=False, drop_last=False)

    results = {}

    for batch, orig_idx in tqdm(zip(loader, indices_sorted), total=len(indices_sorted),
                                 desc="Forward passes"):
        # Single sample (batch_size=1, so j=0 always)
        spec = batch[SPECTRUM][0:1].to(device)
        # RawSpectraDataset returns raw adduct strings, not indices
        # Convert using str_to_idx dict
        adduct_raw = batch[ADDUCT][0] if ADDUCT in batch else None
        if adduct_raw is not None:
            adduct_str = adduct_raw.decode('utf-8') if isinstance(adduct_raw, bytes) else adduct_raw
            adduct_idx = str_to_idx.get(adduct_str, 1)  # 1 = UNK
            adduct_t = torch.tensor([adduct_idx], dtype=torch.long, device=device)
        else:
            adduct_t = None

        ce_val = batch[COLLISION_ENERGY][0:1] if COLLISION_ENERGY in batch else None
        ce_t = torch.as_tensor(ce_val, dtype=torch.float32, device=device) if ce_val is not None else None

        # Register hooks
        hooks, storage = get_attention_hooks(model)

        # Forward
        with torch.inference_mode():
            out = model(spec, charge=None,
                        adduct=adduct_t,
                        collision_energy=ce_t)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Collect results
        prec_mz = batch[PRECURSOR_MZ]
        if isinstance(prec_mz, (list, tuple)):
            prec_mz = prec_mz[0]
        result = {
            'precursor_mz': float(prec_mz),
            'adduct_str': adduct_str if adduct_raw is not None else 'N/A',
            'ce': float(ce_val[0]) if ce_val is not None else 0.0,
            'attention': {},
            'output_shape': out.shape,
        }

        for layer_idx, st in storage.items():
            if 'attn' in st:
                result['attention'][layer_idx] = st['attn']

        results[orig_idx] = result

    return results


def save_attention_heatmaps(results, output_dir, max_layers=7, max_heads=8):
    """Generate and save attention heatmap visualizations (C2).

    Creates:
      - {output_dir}/heatmaps/avg_cond_focus.png — Frag→Adduct/CE across layers×heads
      - {output_dir}/heatmaps/layer{N}_head{H}_idx{IDX}.png — per-sample heatmaps
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import numpy as np
    heat_dir = os.path.join(output_dir, 'heatmaps')
    os.makedirs(heat_dir, exist_ok=True)

    adduct_pos, ce_pos, frag_start = 1, 2, 3

    # ── 1) Per-sample full attention matrix heatmaps ──
    for idx, res in sorted(results.items()):
        n_saved = 0
        for layer_idx in sorted(res['attention'].keys()):
            attn = res['attention'][layer_idx][0]  # (n_heads, S, S)
            S = attn.shape[-1]

            # Token labels
            token_labels = ['Prec', 'Ad', 'CE'] + [f'F{i}' for i in range(3, S)]
            # Show every 5th fragment label to avoid crowding
            labels_show = token_labels[:3] + [''] * (S - 3)
            for i in range(3, S, 5):
                labels_show[i] = f'F{i}'

            # Plot average over heads
            fig, ax = plt.subplots(figsize=(10, 8))
            attn_avg = attn.mean(dim=0).numpy()  # (S, S)
            # Use fixed vmax=0.1: off-diagonal values are typically 0.01-0.05,
            # diagonal self-attention is 0.1-0.3. capping at 0.1 reveals
            # the interesting off-diagonal structure; diagonal saturates.
            vmax = 0.1
            im = ax.imshow(attn_avg, cmap='viridis', aspect='equal',
                           norm=Normalize(vmin=0, vmax=vmax))
            ax.set_xticks(range(S))
            ax.set_yticks(range(S))
            ax.set_xticklabels(labels_show, rotation=90, fontsize=5)
            ax.set_yticklabels(labels_show, fontsize=5)
            ax.set_xlabel('Attended TO →', fontsize=10)
            ax.set_ylabel('Attended FROM ↓', fontsize=10)
            ax.set_title(f'Spectrum idx={idx} ({res["adduct_str"]}, CE={res["ce"]:.0f}) Layer {layer_idx} (avg {max_heads} heads)',
                         fontsize=10)

            # Highlight cond token columns
            for pos, color, label in [(adduct_pos, 'red', 'Ad'), (ce_pos, 'cyan', 'CE')]:
                ax.add_patch(plt.Rectangle((pos-0.5, frag_start-0.5), 1, S-frag_start,
                                            fill=False, edgecolor=color, linewidth=1.5, linestyle='--'))

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fname = os.path.join(heat_dir, f'layer{layer_idx}_avg_idx{idx}_ce{res["ce"]:.0f}.png')
            fig.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close(fig)
            n_saved += 1

            # Plot per-head (only first 4 samples, layers 0 and last)
            if n_saved <= 1 or layer_idx in (0, max(list(res['attention'].keys()) + [0])):
                for h in range(min(max_heads, 4)):
                    fig_h, ax_h = plt.subplots(figsize=(8, 7))
                    attn_h = attn[h].numpy()
                    vmax_h = 0.1
                    ax_h.imshow(attn_h, cmap='viridis', aspect='equal',
                                norm=Normalize(vmin=0, vmax=vmax_h))
                    ax_h.set_xticks(range(S))
                    ax_h.set_yticks(range(S))
                    ax_h.set_xticklabels(labels_show, rotation=90, fontsize=5)
                    ax_h.set_yticklabels(labels_show, fontsize=5)
                    ax_h.set_title(f'idx={idx} L{layer_idx} H{h} ({res["adduct_str"]}, CE={res["ce"]:.0f})', fontsize=9)
                    for pos, color in [(adduct_pos, 'red'), (ce_pos, 'cyan')]:
                        ax_h.add_patch(plt.Rectangle((pos-0.5, frag_start-0.5), 1, S-frag_start,
                                                      fill=False, edgecolor=color, linewidth=1, linestyle='--'))
                    fname_h = os.path.join(heat_dir, f'layer{layer_idx}_head{h}_idx{idx}_ce{res["ce"]:.0f}.png')
                    fig_h.savefig(fname_h, dpi=150, bbox_inches='tight')
                    plt.close(fig_h)

    # ── 2) Cond-focus summary: Frag→Adduct and Frag→CE per layer×head ──
    n_layers = max(list(res['attention'].keys()) for res in results.values())
    n_layers = max(n for n in n_layers) + 1 if n_layers else 7

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for metric_idx, metric_name, target_pos in [
        (0, 'Frag → Adduct', adduct_pos), (1, 'Frag → CE', ce_pos)]:
        # Build matrix: (n_spectra × n_layers × n_heads) → average over spectra
        data = np.zeros((n_layers, max_heads))
        counts = np.zeros((n_layers, max_heads))
        for res in results.values():
            for li in sorted(res['attention'].keys()):
                attn = res['attention'][li][0]  # (n_heads, S, S)
                for h in range(min(max_heads, attn.shape[0])):
                    data[li, h] += attn[h, frag_start:, target_pos].mean().item()
                    counts[li, h] += 1
        data /= np.maximum(counts, 1)

        im = axes[metric_idx].imshow(data, cmap='YlOrRd', aspect='auto',
                                     vmin=0, vmax=data.max()*1.1 if data.max()>0 else 1)
        axes[metric_idx].set_xticks(range(max_heads))
        axes[metric_idx].set_yticks(range(n_layers))
        axes[metric_idx].set_xticklabels([f'H{h}' for h in range(max_heads)])
        axes[metric_idx].set_yticklabels([f'L{li}' for li in range(n_layers)])
        axes[metric_idx].set_xlabel('Head')
        axes[metric_idx].set_ylabel('Layer')
        axes[metric_idx].set_title(metric_name)
        plt.colorbar(im, ax=axes[metric_idx], fraction=0.046)

    plt.tight_layout()
    fname = os.path.join(heat_dir, 'cond_focus_heatmap.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Heatmaps saved to: {heat_dir}/")
    return heat_dir


def analyze_attention(results, output_dir):
    """Compute and save attention analysis."""
    from collections import defaultdict
    os.makedirs(output_dir, exist_ok=True)

    summary_lines = []
    summary_lines.append("=" * 90)
    summary_lines.append("ATTENTION ANALYSIS: Cond Token Model")
    summary_lines.append("=" * 90)

    for idx, res in sorted(results.items()):
        summary_lines.append(f"\nSpectrum idx={idx} | {res['adduct_str']} | "
                             f"CE={res['ce']:.1f} | prec_mz={res['precursor_mz']:.2f} | "
                             f"output shape={res['output_shape']}")

        n_layers = len(res['attention'])
        # S = seq_len (63 for cond model)
        first_attn = list(res['attention'].values())[0]
        B, n_heads, S, _ = first_attn.shape
        summary_lines.append(f"  Layers: {n_layers}, Heads: {n_heads}, Sequence: {S} tokens")

        # ── Per-layer analysis ──
        for layer_idx in sorted(res['attention'].keys()):
            attn = res['attention'][layer_idx]  # (1, n_heads, S, S)
            attn = attn[0]  # (n_heads, S, S)

            # Position indices in the token sequence:
            # 0 = precursor, 1 = adduct, 2 = CE, 3..62 = fragment peaks
            adduct_pos = 1
            ce_pos = 2
            frag_start = 3

            # ── How much do fragment peaks attend to cond tokens? ──
            # Average attention FROM fragment tokens TO adduct/CE tokens
            frag_to_adduct = attn[:, frag_start:, adduct_pos].mean().item()
            frag_to_ce = attn[:, frag_start:, ce_pos].mean().item()
            frag_to_precursor = attn[:, frag_start:, 0].mean().item()
            frag_to_self = attn[:, frag_start:, frag_start:].mean().item()

            # ── How much do cond tokens attend to fragments? ──
            adduct_to_frag = attn[:, adduct_pos, frag_start:].mean().item()
            ce_to_frag = attn[:, ce_pos, frag_start:].mean().item()

            # ── Self-attention of cond tokens ──
            adduct_self = attn[:, adduct_pos, adduct_pos].mean().item()
            ce_self = attn[:, ce_pos, ce_pos].mean().item()
            adduct_to_ce = attn[:, adduct_pos, ce_pos].mean().item()
            ce_to_adduct = attn[:, ce_pos, adduct_pos].mean().item()

            summary_lines.append(
                f"  Layer {layer_idx:2d}: "
                f"Frag→Adduct={frag_to_adduct:.4f} "
                f"Frag→CE={frag_to_ce:.4f} "
                f"Frag→Prec={frag_to_precursor:.4f} "
                f"Frag↔Frag={frag_to_self:.4f} | "
                f"Ad→Frag={adduct_to_frag:.4f} "
                f"CE→Frag={ce_to_frag:.4f}"
            )

            # Also show per-head breakdown for a few heads (most interesting layer)
            if layer_idx == 0 or layer_idx == n_layers - 1:
                summary_lines.append(f"    Per-head Frag→Adduct:")
                for h in range(min(n_heads, 8)):
                    ha = attn[h, frag_start:, adduct_pos].mean().item()
                    hc = attn[h, frag_start:, ce_pos].mean().item()
                    summary_lines.append(f"      Head {h}: Frag→Adduct={ha:.4f}, Frag→CE={hc:.4f}")

        # ── Save full attention matrices as .npy ──
        attn_dict = {}
        for layer_idx, attn_tensor in res['attention'].items():
            attn_dict[f'layer_{layer_idx}'] = attn_tensor.numpy()
        npz_path = os.path.join(output_dir, f'attn_idx{idx}_ce{res["ce"]:.0f}.npz')
        np.savez_compressed(npz_path, **attn_dict)
        summary_lines.append(f"  Saved: {npz_path}")

    # ── Cross-spectrum comparison ──
    summary_lines.append("\n" + "=" * 90)
    summary_lines.append("CROSS-SPECTRUM COMPARISON: Same adduct, different CE")
    summary_lines.append("=" * 90)

    # Group by adduct
    by_adduct = defaultdict(list)
    for idx, res in results.items():
        by_adduct[res['adduct_str']].append((idx, res))

    for adduct_name, spectra_list in sorted(by_adduct.items()):
        if len(spectra_list) >= 2:
            summary_lines.append(f"\n{adduct_name} ({len(spectra_list)} spectra):")
            spectra_list.sort(key=lambda x: x[1]['ce'])

            # Compare Frag→CE attention vs CE value
            header = f"  {'CE':>6} | Frag→CE (layer means):"
            for li in sorted(results[spectra_list[0][0]]['attention'].keys()):
                header += f"  L{li:>2}"
            summary_lines.append(header)
            summary_lines.append("  " + "-" * (len(header)))

            for idx, res in spectra_list:
                line = f"  {res['ce']:>6.0f} |"
                for li in sorted(res['attention'].keys()):
                    attn = res['attention'][li][0]
                    val = attn[:, 3:, 2].mean().item()
                    line += f"  {val:.4f}"
                summary_lines.append(line)

    # Save summary
    summary_path = os.path.join(output_dir, 'attention_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nSummary saved to: {summary_path}")
    print('\n'.join(summary_lines[-20:]))

    # ── C2: Generate heatmaps ──
    save_attention_heatmaps(results, output_dir)

    return summary_path


def main():
    parser = argparse.ArgumentParser(description='Attention analysis for cond-token DreaMS')
    parser.add_argument('--ckpt', default='/root/DreaMS/dreams_cond/v1/last-v1.ckpt',
                        help='Model checkpoint path')
    parser.add_argument('--hdf5', default='/root/datasets/dreams_ready.hdf5',
                        help='HDF5 dataset path')
    parser.add_argument('--output', default='/root/DreaMS/attention_results',
                        help='Output directory for results')
    parser.add_argument('--indices', type=str, default=None,
                        help='Comma-separated spectrum indices (default: [M+H]+ at various CE)')
    parser.add_argument('--n_samples', type=int, default=12,
                        help='Number of spectra to analyze (default: 12)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device')
    args = parser.parse_args()

    print("=" * 90)
    print("COND-TOKEN ATTENTION ANALYSIS")
    print("=" * 90)

    # ── Load model ──
    print("\n[1] Loading model...")
    model, str_to_idx = load_cond_model(args.ckpt, args.hdf5, device=args.device)

    # ── Load HDF5 data ──
    print("\n[2] Loading HDF5 data...")
    msdata = MSData.load(args.hdf5)
    cols = msdata.columns()
    print(f"  Columns: {cols}")
    print(f"  Total spectra: {len(msdata)}")

    # ── Select spectra for analysis ──
    print("\n[3] Selecting spectra...")
    if args.indices:
        indices = [int(i.strip()) for i in args.indices.split(',')]
    else:
        # Auto-select [M+H]+ spectra across CE range with same precursor m/z = 360
        import h5py
        with h5py.File(args.hdf5, 'r') as f:
            adducts = f['adduct'][:]
            ce = f['collision_energy'][:]
            prec_mz = f['precursor_mz'][:]

        # Decode adducts
        adduct_strs = np.array([a.decode() if isinstance(a, bytes) else a for a in adducts])

        # [M+H]+ at various CE with same precursor
        mh_mask = adduct_strs == '[M+H]+'
        mh_idxs = np.where(mh_mask)[0]

        # Pick same precursor m/z = 360 at various CEs
        same_prec = mh_idxs[np.abs(prec_mz[mh_idxs] - 360.0) < 1.0]
        # Sort by CE
        same_prec_sorted = same_prec[np.argsort(ce[same_prec])]
        # Take diverse CE values
        indices = []
        for target in [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 100, 120]:
            closest = same_prec_sorted[np.argmin(np.abs(ce[same_prec_sorted] - target))]
            if closest not in indices:
                indices.append(int(closest))

        # Also add [M+Na]+, [M-H]-, and rare adducts
        for ad_name in ['[M+Na]+', '[M-H]-', '[M+K]+', '[2M+H]+']:
            mask = adduct_strs == ad_name
            idxs = np.where(mask)[0]
            if len(idxs) > 0:
                # Pick the one with CE closest to 35
                best = idxs[np.argmin(np.abs(ce[idxs] - 35))]
                if best not in indices:
                    indices.append(int(best))

        indices = indices[:args.n_samples]

    print(f"  Analyzing {len(indices)} spectra at indices: {indices}")

    # Print details
    import h5py
    with h5py.File(args.hdf5, 'r') as f:
        adducts_all = f['adduct'][:]
        ce_all = f['collision_energy'][:]
        prec_mz_all = f['precursor_mz'][:]
    adduct_strs_all = np.array([a.decode() if isinstance(a, bytes) else a for a in adducts_all])
    for idx in indices:
        print(f"    idx={idx:>6}: {adduct_strs_all[idx]:>15}  CE={ce_all[idx]:>3.0f}  "
              f"prec_mz={prec_mz_all[idx]:.2f}")

    # ── Run attention analysis ──
    print("\n[4] Running forward passes with attention hooks...")
    results = run_attention_on_spectra(
        model, msdata, set(indices), str_to_idx,
        device=args.device, batch_size=1
    )

    # ── Analyze ──
    print("\n[5] Analyzing attention patterns...")
    out_dir = analyze_attention(results, args.output)

    print(f"\nDone! Results saved to: {out_dir}")
    print(f"  Attention matrices: {out_dir}/attn_idx*.npz")
    print(f"  Summary:           {out_dir}/attention_summary.txt")


if __name__ == '__main__':
    main()
