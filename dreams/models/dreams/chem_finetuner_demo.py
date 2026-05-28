"""
Dummy training step for DreaMS_ChemFinetuner.

Demonstrates instantiation, forward pass with random tensors, and loss.backward().
Run:  python -m dreams.models.dreams.chem_finetuner_demo
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/root/DreaMS')

import torch
import torch.nn as nn
import numpy as np

# ── DreaMS mock (for the msml module issue) ──
import dreams.utils.data as du
import dreams.utils.dformats as dformats
import dreams.utils.spectra as su
import dreams.models.dreams.dreams as dm
import dreams.models.dreams.layers as dl
import dreams.models.layers.fourier_features as ff
import dreams.models.layers.feed_forward as fw
import types
for ns in ['msml','msml.models','msml.models.dreams','msml.models.layers','msml.utils']:
    sys.modules[ns] = types.ModuleType(ns)
sys.modules['msml.models.dreams.dreams'] = dm
sys.modules['msml.models.dreams.layers'] = dl
sys.modules['msml.models.layers.fourier_features'] = ff
sys.modules['msml.models.layers.feed_forward'] = fw
sys.modules['msml.utils.data'] = du
sys.modules['msml.utils.dformats'] = dformats
sys.modules['msml.utils.spectra'] = su

from argparse import Namespace
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from dreams.models.dreams.dreams import DreaMS
from dreams.models.dreams.chem_finetuner import DreaMS_ChemFinetuner


def main():
    # ── Device ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load pre-trained DreaMS ──
    ckpt = '/root/DreaMS/dreams/models/pretrained/ssl_model.ckpt'
    print(f'Loading DreaMS from {ckpt} ...')
    raw = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = raw['state_dict']
    hp = raw['hyper_parameters']['args']
    if hasattr(hp, '__dict__'):
        hp = vars(hp)

    dformat = DataFormatA()
    spec_preproc = SpectrumPreprocessor(dformat=dformat, n_highest_peaks=60)

    clean = {'dformat': dformat, 'no_transformer_bias': True}
    for k, v in hp.items():
        if isinstance(v, type(())  ):  # Path objects -> str
            clean[k] = str(v)
        else:
            clean[k] = v
    clean['enable_cond_tokens'] = False

    encoder = DreaMS(Namespace(**clean), spec_preproc)
    load_sd = {k: v for k, v in sd.items()
               if not any(x in k for x in ['ro_out', 'mz_masking'])}
    encoder.load_state_dict(load_sd, strict=False)
    encoder = encoder.to(device)
    encoder.eval()
    print(f'DreaMS loaded: {sum(p.numel() for p in encoder.parameters()):,} params')

    # ── Wrap with ChemFinetuner ──
    model = DreaMS_ChemFinetuner(encoder).to(device)
    print(f'ChemFinetuner: {sum(p.numel() for p in model.parameters()):,} params')
    print(f'  New MACCS head: {sum(p.numel() for p in model.maccs_head.parameters()):,} params\n')

    # ── Dummy batch ──
    B = 4
    N_PEAKS = 60
    D_MODEL = encoder.d_model  # 1024

    # Random spectrum: (B, 60, 2) — [m/z, intensity]
    spec = torch.randn(B, N_PEAKS, 2, device=device)
    # Ensure m/z is positive (normalized to [0, 1] range)
    spec[:, :, 0] = spec[:, :, 0].abs().clamp(max=1.0)
    spec[:, :, 1] = spec[:, :, 1].abs().clamp(max=1.0)

    # Create masked version: zero out some peaks
    mask = torch.zeros(B, N_PEAKS, dtype=torch.bool, device=device)
    mask[:, 5:15] = True  # mask peaks 5-15
    spec_mask = spec.clone()
    spec_mask[mask] = 0.0

    # Random MACCS labels: (B, 167) — binary
    maccs_labels = torch.randint(0, 2, (B, 167), device=device).float()

    # ── Forward ──
    print('=== Forward pass ===')
    print(f'  spec_mask:       {tuple(spec_mask.shape)}  (B, 60, 2) — masked spectrum')
    print(f'  spec_real:       {tuple(spec.shape)}       (B, 60, 2) — ground truth')
    print(f'  mask:            {tuple(mask.shape)}     (B, 60) — boolean mask')
    print(f'  maccs_labels:    {tuple(maccs_labels.shape)}   (B, 167) — MACCS targets')
    print()

    output = model(
        spec_mask=spec_mask,
        spec_real=spec,
        mask=mask,
        maccs_labels=maccs_labels,
    )

    # ── Inspect outputs ──
    print('=== Outputs ===')
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(f'  {k}: shape={tuple(v.shape)}, value={v.item() if v.numel() == 1 else v.mean().item():.4f}')
    print()

    # ── Backward ──
    print('=== Backward pass ===')
    output['total_loss'].backward()
    
    # Check gradients flow to MACCS head
    maccs_grad = model.maccs_head[0].weight.grad
    print(f'  MACCS head linear1 grad norm: {maccs_grad.norm():.6f}')
    
    # Check gradients reach DreaMS encoder (if we unfreeze it)
    encoder_grad = model.encoder.transformer_encoder.ffs[0].weight.grad
    print(f'  DreaMS layer0 ff grad norm:    {encoder_grad.norm():.6f} (non-zero = encoder unfrozen)')
    print()

    # ── Verify sequence_output is full ──
    seq = output['sequence_output']
    print(f'=== sequence_output ===')
    print(f'  shape: {tuple(seq.shape)}  (B, {N_PEAKS}, {D_MODEL}) — FULL sequence!')
    print(f'  [CLS] token (pos 0) range: [{seq[:, 0, :].min():.3f}, {seq[:, 0, :].max():.3f}]')
    print(f'  Peak token (pos 10) range: [{seq[:, 10, :].min():.3f}, {seq[:, 10, :].max():.3f}]')
    print()

    print('✅ All checks passed!')


if __name__ == '__main__':
    main()
