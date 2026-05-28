"""Predict with encoder-decoder architecture (T5 / MolT5 / BioT5).

Usage:
    python /root/DreaMS/ms2mol_encdec/predict.py --checkpoint /path/to/best.ckpt
    python /root/DreaMS/ms2mol_encdec/predict.py --checkpoint /path/to/best.ckpt \\
        --model_name molt5-small
"""

import argparse
import os
import torch
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/root/DreaMS')

from ms2mol_encdec.model import MSToSMILES_T5, info_display, MODEL_REGISTRY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--model_name', default='t5-small',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Backbone model matching the checkpoint')
    parser.add_argument('--ms_embedding', type=str, default=None,
                        help='Path to .pt file with MS embedding')
    parser.add_argument('--smiles', type=str, default=None,
                        help='SMILES string for reference (optional)')
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--k_tokens', type=int, default=16,
                        help='Number of prefix tokens (must match checkpoint)')
    parser.add_argument('--projector_type', default=None,
                        choices=['mlp', 'k_heads'],
                        help='Override checkpoint projector type')
    parser.add_argument('--projector_depth', type=int, default=None,
                        help='Projector MLP depth (mlp only; default from checkpoint)')
    parser.add_argument('--projector_trunk_dim', type=int, default=None)
    parser.add_argument('--projector_head_rank', type=int, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint.get('model_name', args.model_name)
    k_tokens = checkpoint.get('k_tokens', args.k_tokens)
    projector_type = args.projector_type or checkpoint.get('projector_type', 'mlp')
    projector_depth = args.projector_depth if args.projector_depth is not None else checkpoint.get('projector_depth', 2)
    projector_trunk_dim = args.projector_trunk_dim or checkpoint.get('projector_trunk_dim', 512)
    projector_head_rank = args.projector_head_rank or checkpoint.get('projector_head_rank', 64)

    # Load model
    display = info_display(model_name)
    print(f'Loading {display}...')
    model = MSToSMILES_T5(
        k_tokens=k_tokens, model_name=model_name,
        projector_type=projector_type,
        projector_depth=projector_depth,
        projector_trunk_dim=projector_trunk_dim,
        projector_head_rank=projector_head_rank,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    print(f'Loaded checkpoint: epoch {checkpoint.get("epoch", "?")}, '
          f'val_loss={checkpoint.get("val_loss", 0):.4f}')

    # Get MS embedding
    if args.ms_embedding:
        ms_emb = torch.load(args.ms_embedding, map_location=device)
    else:
        # Random embedding for testing
        ms_emb = torch.randn(1, 1024, device=device)

    # Generate
    generated = model.generate(ms_emb, num_beams=args.num_beams, device=device)

    print(f'\nGenerated SMILES: {generated[0]}')
    if args.smiles:
        print(f'Reference SMILES: {args.smiles}')


if __name__ == '__main__':
    main()
