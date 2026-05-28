"""Predict with RAG architecture.

Usage:
    python /root/DreaMS/ms2mol_rag/predict.py --checkpoint /path/to/best.ckpt --smiles "c1ccccc1"
"""

import argparse
import os
import torch
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/root/DreaMS')

from ms2mol_rag.model import MSToSMILES_RAG, RAGIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--rag_index', type=str, default=None,
                        help='Path to RAG index .pkl')
    parser.add_argument('--ms_embedding', type=str, default=None)
    parser.add_argument('--smiles', type=str, default=None)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--show_context', action='store_true', default=False)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    model = MSToSMILES_RAG(k_tokens=4, k_contexts=3).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # Load RAG index
    if args.rag_index:
        index = RAGIndex()
        index.load(args.rag_index)
        model.set_index(index)

    print(f'Loaded checkpoint: epoch {checkpoint.get("epoch", "?")}, val_loss={checkpoint.get("val_loss", "?"):.4f}')

    # Get MS embedding
    if args.ms_embedding:
        ms_emb = torch.load(args.ms_embedding, map_location=device)
    else:
        ms_emb = torch.randn(1, 1024, device=device)

    # Retrieve context
    context = None
    if model.index is not None:
        ctx_smiles, ctx_dists = model.index.search(ms_emb.squeeze(0), exclude_self=False)
        context = [ctx_smiles]
        if args.show_context:
            print(f'\nRetrieved context molecules:')
            for i, (smi, dist) in enumerate(zip(ctx_smiles, ctx_dists)):
                print(f'  {i+1}. {smi} (dist={dist:.4f})')

    # Generate
    generated = model.generate(ms_emb, context_smiles=context,
                                num_beams=args.num_beams, device=device)

    print(f'\nGenerated SMILES: {generated[0]}')
    if args.smiles:
        print(f'Reference SMILES: {args.smiles}')


if __name__ == '__main__':
    main()
