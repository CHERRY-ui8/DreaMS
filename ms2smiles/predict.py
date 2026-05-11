"""Inference: MS embedding → SMILES generation.

Usage:
    # From precomputed embedding
    conda run -n dreams python -m ms2smiles.predict --ckpt outputs/best.ckpt \\
        --embedding_idx 0 --split test

    # From raw spectrum (e2e mode)
    conda run -n dreams python -m ms2smiles.predict --ckpt outputs/best.ckpt \\
        --spectrum_idx 0 --split test --raw_hdf5 /root/datasets/pairs_ready.hdf5
"""

import argparse
import json
from pathlib import Path

import torch
import h5py
import numpy as np

from ms2smiles.config import MS2SMILESConfig
from ms2smiles.model import MStoSMILES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint path')
    parser.add_argument('--model_size', default='19M', choices=['19M', '1.2B'])

    # Input: either precomputed embedding or raw spectrum
    parser.add_argument('--embedding_idx', type=int, default=None,
                        help='index in pairs_with_embs.hdf5')
    parser.add_argument('--spectrum_idx', type=int, default=None,
                        help='index in pairs_ready.hdf5')
    parser.add_argument('--embedding_path', type=str, default=None,
                        help='path to .npy file with single embedding')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])

    # Full data files
    parser.add_argument('--embs_hdf5', default='/root/datasets/pairs_with_embs.hdf5')
    parser.add_argument('--raw_hdf5', default='/root/datasets/pairs_ready.hdf5')

    # Generation
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=10,
                        help='number of samples to generate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def load_from_embs_hdf5(path, indices, split='test'):
    """Load embeddings + SMILES from pairs_with_embs.hdf5 by split and indices."""
    with h5py.File(path, 'r') as f:
        split_data = f['split'][:]
        mask = split_data == {'train': 0, 'val': 1, 'test': 2}[split]

        all_embs = f['embedding'][:]
        all_smiles = f['smiles'][:]

        # Get all indices in this split
        split_indices = np.where(mask)[0]

        if indices is None:
            indices = split_indices[:args.num_samples]
        elif isinstance(indices, int):
            indices = [indices]

        embs = [torch.from_numpy(all_embs[i]).float() for i in indices]
        smiles = [s.decode() if isinstance(s, bytes) else s for s in all_smiles[indices]]

    return torch.stack(embs), smiles


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f'Device: {device}')

    # Config
    config = MS2SMILESConfig(model_size=args.model_size)
    print(f'Model: ChemGPT-{args.model_size}')

    # Load model
    print(f'Loading checkpoint: {args.ckpt}')
    model = MStoSMILES(config, device=device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()
    print(f'Loaded epoch {state.get("epoch", "?")}, val_loss={state.get("val_loss", "?"):.4f}')

    # Get embeddings
    if args.embedding_path:
        emb = torch.from_numpy(np.load(args.embedding_path)).float().to(device)
        embeddings = emb.unsqueeze(0) if emb.dim() == 1 else emb
        ref_smiles = None
    elif args.embedding_idx is not None or args.spectrum_idx is not None:
        if args.embedding_idx is not None:
            embeddings, ref_smiles_list = load_from_embs_hdf5(
                args.embs_hdf5, args.embedding_idx, args.split,
            )
        else:
            # Load raw spectrum and compute embedding (e2e)
            with h5py.File(args.raw_hdf5, 'r') as f:
                split_data = f['split'][:]
                mask = split_data == {'train': 0, 'val': 1, 'test': 2}[args.split]
                if args.spectrum_idx is not None:
                    idx = np.where(mask)[0][args.spectrum_idx]
                    spectrum = torch.from_numpy(f['spectrum'][idx]).float().unsqueeze(0)
                    sm = f['smiles'][idx].decode() if isinstance(f['smiles'][idx], bytes) else f['smiles'][idx]
                    ref_smiles_list = [sm]
                    embeddings = model.extract_ms_embedding(spectrum)
        ref_smiles = ref_smiles_list[0] if ref_smiles_list else None
    else:
        # Load first N samples from test set
        print(f'Loading {args.num_samples} samples from {args.split} set...')
        with h5py.File(args.embs_hdf5, 'r') as f:
            split_data = f['split'][:]
            mask = split_data == {'train': 0, 'val': 1, 'test': 2}[args.split]
            split_idx = np.where(mask)[0]
            selected = split_idx[:args.num_samples]

            embs_arr = f['embedding'][selected]
            smiles_arr = f['smiles'][selected]

        embeddings = torch.from_numpy(embs_arr).float()
        ref_smiles_list = [s.decode() if isinstance(s, bytes) else s for s in smiles_arr]
        ref_smiles = ref_smiles_list[0] if ref_smiles_list else None

    embeddings = embeddings.to(device)
    print(f'Input embeddings: {embeddings.shape}')

    # Generate
    print(f'\nGenerating SMILES (beam={args.num_beams}, max_len={args.max_length})...\n')
    generated = model.generate(
        embeddings=embeddings,
        max_length=args.max_length,
        num_beams=args.num_beams,
    )

    # Print results
    if ref_smiles_list and len(ref_smiles_list) > 0:
        print(f'{"#"*60}')
        print(f'Reference: {ref_smiles_list[0]}')
        print(f'Generated: {generated[0]}')
        print(f'{"#"*60}')
    else:
        for i, smi in enumerate(generated):
            print(f'[{i}] {smi}')

    # If multiple samples, print table
    if len(generated) > 1:
        print(f'\n--- Results ({len(generated)} samples) ---')
        from rdkit import Chem
        valid = 0
        for i, (smi, ref) in enumerate(zip(generated, ref_smiles_list)):
            mol = Chem.MolFromSmiles(smi)
            is_valid = mol is not None
            if is_valid:
                valid += 1
            match = '✓' if smi == ref else ('≈' if is_valid and Chem.MolToSmiles(mol) == Chem.MolToSmiles(
                Chem.MolFromSmiles(ref)) else '✗')
            print(f'  [{i}] ref={ref}')
            print(f'       gen={smi}  {"✓ valid" if is_valid else "✗ invalid"} {match}')

        print(f'\nValid SMILES rate: {valid}/{len(generated)} ({100*valid/len(generated):.1f}%)')

    print('\nDone!')


if __name__ == '__main__':
    main()
