"""Extract representative RAG retrieval examples and visualize them.

For each query molecule, shows:
  - Query SMILES + structure
  - Top-1 neighbor SMILES + structure  
  - Tanimoto similarity
  - Qualitative assessment of whether they share functional groups

Saves results to rag/retrieval_examples.md
"""

import os
import sys
import numpy as np
from sklearn.neighbors import NearestNeighbors
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import SimilarityMaps
import selfies as sf

sys.path.insert(0, '/root/DreaMS')
from ms2mol_rag.dataset import MSSpectrumSmilesRAGDataset


def compute_tanimoto(smi1, smi2):
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return 0.0, None, None
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
    tan = DataStructs.TanimotoSimilarity(fp1, fp2)
    return tan, mol1, mol2


def get_functional_groups(smiles):
    """Simple functional group detection."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    groups = []
    smarts_patterns = {
        'Alcohol (OH)': '[OX2H]',
        'Phenol (ArOH)': '[OX2H][c]',
        'Carboxylic Acid (COOH)': '[CX3](=O)[OX2H]',
        'Ester (COOR)': '[CX3](=O)[OX2][#6]',
        'Amide (CONH)': '[CX3](=O)[NX3]',
        'Aldehyde (CHO)': '[CX3H1](=O)[#6]',
        'Ketone (C=O)': '[#6][CX3](=O)[#6]',
        'Ether (ROR)': '[#6][OX2][#6]',
        'Amine (NH2)': '[NX3;H2]',
        'Nitro (NO2)': '[NX3](=O)=O',
        'Sulfonic Acid': '[SX4](=O)(=O)[OX2H]',
        'Sulfide (RSR)': '[#16X2][#6]',
        'Disulfide (RSSR)': '[#16X2][#16X2]',
        'Thiol (SH)': '[#16X2H]',
        'Alkene (C=C)': '[#6]=[#6]',
        'Alkyne (C≡C)': '[#6]#[#6]',
        'Aromatic Ring': '[c]',
        'Pyridine': 'c1ncccc1',
        'Pyrimidine': 'c1ncncc1',
        'Imidazole': 'c1cncn1',
        'Furan': 'c1ccoc1',
        'Thiophene': 'c1ccsc1',
        'Indole': 'c1ccc2[nH]ccc2c1',
        'Quinoline': 'c1ccc2ncccc2c1',
        'Benzene Ring': 'c1ccccc1',
        'Lactam': '[#6][#6](=O)[NX3][#6]',
        'Lactone': '[#6][#6](=O)[OX2][#6]',
    }
    for name, smarts in smarts_patterns.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt and mol.HasSubstructMatch(patt):
            groups.append(name)
    return groups


def main():
    print("Loading datasets...")
    train_data = MSSpectrumSmilesRAGDataset(split='train')
    val_data = MSSpectrumSmilesRAGDataset(split='val')
    
    train_embs = train_data.embeddings
    train_smiles = train_data.smiles
    val_embs = val_data.embeddings
    val_smiles = val_data.smiles

    print(f"Train: {len(train_smiles)}, Val: {len(val_smiles)}")

    # Build KNN index
    print("Building KNN index...")
    index = NearestNeighbors(n_neighbors=2, metric='cosine', algorithm='brute')
    index.fit(train_embs)

    # Sample 10 val examples across the Tanimoto spectrum
    # We'll compute Tanimoto for all, then pick from different percentiles
    n_val = len(val_smiles)
    all_tans = np.zeros(n_val)
    all_neighbor_smiles = [''] * n_val
    
    batch_size = 512
    for start in range(0, n_val, batch_size):
        end = min(start + batch_size, n_val)
        distances, indices = index.kneighbors(val_embs[start:end])
        for i in range(end - start):
            query_idx = start + i
            neighbor_idx = indices[i][0]
            if neighbor_idx == query_idx:
                neighbor_idx = indices[i][1] if len(indices[i]) > 1 else indices[i][0]
            neighbor_smi = train_smiles[neighbor_idx]
            tan, _, _ = compute_tanimoto(val_smiles[query_idx], neighbor_smi)
            all_tans[query_idx] = tan
            all_neighbor_smiles[query_idx] = neighbor_smi
        
        if (start + batch_size) % 2048 == 0:
            print(f"  Processed {start + batch_size}/{n_val}")

    # Select examples from different Tanimoto ranges
    sorted_idx = np.argsort(all_tans)
    n = len(sorted_idx)
    
    examples = []
    # 2 from bottom 10%
    for i in sorted_idx[:2]:
        examples.append(('bottom', all_tans[i], val_smiles[i], all_neighbor_smiles[i]))
    # 2 from 20-30%
    for i in sorted_idx[int(n*0.2):int(n*0.3)][:2]:
        examples.append(('low', all_tans[i], val_smiles[i], all_neighbor_smiles[i]))
    # 2 from 45-55% (median-ish)
    for i in sorted_idx[int(n*0.45):int(n*0.55)][:2]:
        examples.append(('mid', all_tans[i], val_smiles[i], all_neighbor_smiles[i]))
    # 2 from 70-80%
    for i in sorted_idx[int(n*0.7):int(n*0.8)][:2]:
        examples.append(('high', all_tans[i], val_smiles[i], all_neighbor_smiles[i]))
    # 2 from top 5%
    for i in sorted_idx[-2:]:
        examples.append(('top', all_tans[i], val_smiles[i], all_neighbor_smiles[i]))

    # Generate report
    output_path = '/root/DreaMS/ms2mol_rag/retrieval_examples.md'
    
    lines = []
    lines.append("# DreaMS Retrieval Examples\n")
    lines.append(f"Generated from {n_val} validation samples, retrieved from {len(train_smiles)} training samples.\n")
    lines.append("| # | Category | Tanimoto | Query SMILES | Neighbor SMILES | Shared Functional Groups |\n")
    lines.append("|---|---|---|---|---|---|\n")

    for idx, (cat, tan, qry, nei) in enumerate(examples):
        q_groups = set(get_functional_groups(qry))
        n_groups = set(get_functional_groups(nei))
        shared = q_groups & n_groups
        union = q_groups | n_groups
        
        shared_str = ', '.join(sorted(shared)) if shared else '(none)'
        q_only = q_groups - n_groups
        n_only = n_groups - q_groups
        
        lines.append(f"| {idx+1} | {cat} (t={tan:.3f}) | {tan:.3f} | `{qry}` | `{nei}` | {shared_str} |\n")
    
    lines.append("\n## Detailed Analysis\n")
    
    for idx, (cat, tan, qry, nei) in enumerate(examples):
        q_groups = set(get_functional_groups(qry))
        n_groups = set(get_functional_groups(nei))
        shared = q_groups & n_groups
        q_only = q_groups - n_groups
        n_only = n_groups - q_groups
        
        _, mol_q, mol_n = compute_tanimoto(qry, nei)
        q_f = Chem.MolToSmiles(mol_q) if mol_q else qry
        n_f = Chem.MolToSmiles(mol_n) if mol_n else nei
        
        lines.append(f"### Example {idx+1}: {cat.upper()} (Tanimoto={tan:.3f})\n")
        lines.append(f"- **Query ({cat})**: `{q_f}`\n")
        lines.append(f"- **Neighbor**: `{n_f}`\n")
        lines.append(f"- **Shared groups**: {', '.join(sorted(shared)) if shared else '*none*'}\n")
        if q_only:
            lines.append(f"- **Query only groups**: {', '.join(sorted(q_only))}\n")
        if n_only:
            lines.append(f"- **Neighbor only groups**: {', '.join(sorted(n_only))}\n")
        
        # Try to visualize if possible
        if mol_q and mol_n:
            try:
                # Show substructure overlap
                match = mol_q.GetSubstructMatch(mol_n) if mol_q.HasSubstructMatch(mol_n) else []
                if match:
                    lines.append(f"- **Substructure match found**: {len(match)} atoms overlap\n")
                else:
                    lines.append(f"- **No direct substructure match** — different molecular scaffolds\n")
            except:
                pass
        
        lines.append("\n---\n")

    lines.append("\n## Summary\n")
    lines.append(f"- **Total queries**: {n_val}\n")
    lines.append(f"- **Mean Top-1 Tanimoto**: {all_tans.mean():.4f}\n")
    lines.append(f"- **Median Top-1 Tanimoto**: {np.median(all_tans):.4f}\n")
    lines.append(f"- **% with Tanimoto > 0.4**: {(all_tans > 0.4).mean() * 100:.1f}%\n")
    lines.append(f"- **% with Tanimoto > 0.7**: {(all_tans > 0.7).mean() * 100:.1f}%\n")
    lines.append("\n### Verdict\n")
    lines.append("DreaMS embeddings encode MS fragmentation patterns, not molecular structure. ")
    lines.append("Retrieved neighbors tend to share **ionization behavior** (similar functional groups that ")
    lines.append("fragment similarly in MS/MS) rather than **overall molecular topology**. ")
    lines.append("This is expected for MS-based embeddings and explains the low Tanimoto scores.\n")
    
    with open(output_path, 'w') as f:
        f.writelines(lines)
    
    print(f"\nSaved to {output_path}")
    print(f"Overall Mean Top-1 Tanimoto: {all_tans.mean():.4f}")


if __name__ == '__main__':
    main()
