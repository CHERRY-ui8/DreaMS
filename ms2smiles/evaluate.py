"""Evaluate MS→SMILES model: generate samples from test set and compute metrics."""
import sys
sys.path.insert(0, '/root/DreaMS')
import json, warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from ms2smiles.config import MS2SMILESConfig
from ms2smiles.model import MStoSMILES
from ms2smiles.dataset import MSSpectrumSmilesDataset


def tanimoto_similarity(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = '/root/DreaMS/ms2smiles/outputs/test_run/best.ckpt'
    num_samples = 100

    print(f'Device: {device}')
    print(f'Checkpoint: {ckpt_path}')

    config = MS2SMILESConfig(model_size='19M')
    model = MStoSMILES(config, device=device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    # Load test dataset
    dataset = MSSpectrumSmilesDataset(config, split='test')
    n_test = min(num_samples, len(dataset))

    # Batch generate for speed
    all_embs = []
    for i in range(n_test):
        all_embs.append(dataset[i]['embedding'])
    all_embs = torch.stack(all_embs).to(device)  # (N, 1024)

    print(f'Generating {n_test} samples in batches...')
    batch_size = 64
    all_generated = []
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        gen = model.generate(
            embeddings=all_embs[start:end],
            num_beams=5,
            max_length=200,
        )
        all_generated.extend(gen)
        print(f'  [{end}/{n_test}] done')

    # Evaluate
    n_valid = 0
    n_exact = 0
    tanimoto_scores = []

    for i in range(n_test):
        ref_smi = dataset.smiles[i]
        gen_smi = all_generated[i]

        mol_gen = Chem.MolFromSmiles(gen_smi)
        mol_ref = Chem.MolFromSmiles(ref_smi)
        is_valid = mol_gen is not None

        if is_valid:
            n_valid += 1
            gen_canonical = Chem.MolToSmiles(mol_gen)
            ref_canonical = Chem.MolToSmiles(mol_ref)
            if gen_canonical == ref_canonical:
                n_exact += 1

            tan = tanimoto_similarity(ref_smi, gen_smi)
            if tan is not None:
                tanimoto_scores.append(tan)

    # Print results
    print('\n' + '=' * 60)
    print('EVALUATION RESULTS (1 epoch, ChemGPT-19M)')
    print('=' * 60)
    print(f'Samples:           {n_test}')
    print(f'Valid SMILES:      {n_valid}/{n_test} ({100*n_valid/n_test:.1f}%)')
    print(f'Exact match:       {n_exact}/{n_test} ({100*n_exact/n_test:.4f}%)')
    if tanimoto_scores:
        print(f'Tanimoto (mean):   {np.mean(tanimoto_scores):.4f}')
        print(f'Tanimoto (median): {np.median(tanimoto_scores):.4f}')
        print(f'Tanimoto (max):    {max(tanimoto_scores):.4f}')

    # Show sample outputs
    print('\n--- Sample outputs (first 10) ---')
    for i in range(min(10, n_test)):
        ref = dataset.smiles[i]
        gen = all_generated[i]
        mol_g = Chem.MolFromSmiles(gen)
        v = '✓' if mol_g is not None else '✗'
        e = '✓' if v == '✓' and Chem.MolToSmiles(mol_g) == Chem.MolToSmiles(Chem.MolFromSmiles(ref)) else '✗'
        t = ''
        if v == '✓' and e == '✗':
            t_val = tanimoto_similarity(ref, gen)
            t = f' tanimoto={t_val:.3f}' if t_val else ''
        print(f'  [{i}] ref={ref}')
        print(f'       gen={gen}  {v}{e}{t}')

    # Save
    output_path = '/root/DreaMS/ms2smiles/outputs/test_run/eval_results.json'
    results = [{'idx': i, 'reference': dataset.smiles[i], 'generated': all_generated[i],
                'valid': Chem.MolFromSmiles(all_generated[i]) is not None} for i in range(n_test)]
    with open(output_path, 'w') as f:
        json.dump({
            'metrics': {
                'n_samples': n_test,
                'valid_rate': round(n_valid / n_test, 4),
                'exact_match_rate': round(n_exact / n_test, 6),
                'tanimoto_mean': round(float(np.mean(tanimoto_scores)), 4) if tanimoto_scores else None,
                'tanimoto_median': round(float(np.median(tanimoto_scores)), 4) if tanimoto_scores else None,
            },
            'results': results,
        }, f, indent=2)
    print(f'\nResults saved to {output_path}')


if __name__ == '__main__':
    main()
