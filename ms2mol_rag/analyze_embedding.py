"""Analyze DreaMS embedding space: does MS similarity imply structural similarity?

Computes, for each sample in the validation set:
  1. Top-3 most similar MS embeddings (cosine distance) from the training set
  2. Tanimoto similarity between each retrieved molecule and the query molecule
  3. Average Tanimoto across all validation samples

If avg Tanimoto < 0.4, the RAG assumption is weak — DreaMS embeddings
don't reliably encode structural similarity, and contrastive learning
may be needed before RAG is effective.
"""

import os
import sys
import time
import numpy as np
from sklearn.neighbors import NearestNeighbors
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

sys.path.insert(0, '/root/DreaMS')
from ms2mol_rag.dataset import MSSpectrumSmilesRAGDataset


def compute_tanimoto(smi1: str, smi2: str) -> float:
    """Compute Morgan Tanimoto similarity between two SMILES."""
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def main():
    print("=" * 60)
    print("DreaMS Embedding Space Analysis")
    print("=" * 60)
    start_time = time.time()

    # ── Load training and validation data ──
    print("\nLoading training set (for index)...")
    train_data = MSSpectrumSmilesRAGDataset(split='train')
    train_embs = train_data.embeddings  # (N_train, 1024)
    train_smiles = train_data.smiles

    print("\nLoading validation set (for queries)...")
    val_data = MSSpectrumSmilesRAGDataset(split='val')
    val_embs = val_data.embeddings  # (N_val, 1024)
    val_smiles = val_data.smiles

    print(f"\n  Train: {len(train_smiles)} molecules")
    print(f"  Val:   {len(val_smiles)} molecules")

    # ── Build KNN index on training set ──
    print("\nBuilding KNN index (cosine distance)...")
    index = NearestNeighbors(
        n_neighbors=4,  # 4 = 3 neighbors + 1 self (excluded)
        metric='cosine',
        algorithm='brute',
    )
    index.fit(train_embs)
    print("  Done.")

    # ── For each val sample, retrieve top-3 and compute Tanimoto ──
    print("\nRetrieving top-3 neighbors for each validation sample...")
    print("  (This may take a while for 18K samples)\n")

    all_tanimotos = []  # list of [t1, t2, t3] for each query
    n_processed = 0

    batch_size = 512
    n_val = len(val_embs)

    for batch_start in range(0, n_val, batch_size):
        batch_end = min(batch_start + batch_size, n_val)
        batch_embs = val_embs[batch_start:batch_end]

        # Query: find 4 nearest neighbors (3 + self)
        distances, indices = index.kneighbors(batch_embs)

        for i in range(len(batch_embs)):
            query_idx = batch_start + i
            query_smi = val_smiles[query_idx]

            # Skip self (first neighbor is always self if query in train set)
            neighbors = []
            for j, idx in enumerate(indices[i]):
                if idx != query_idx:  # skip self
                    neighbor_smi = train_smiles[idx]
                    tan = compute_tanimoto(query_smi, neighbor_smi)
                    neighbors.append(tan)
                if len(neighbors) == 3:
                    break

            all_tanimotos.append(neighbors)
            n_processed += 1

        elapsed = time.time() - start_time
        print(f'  [{batch_end}/{n_val}] processed, {elapsed:.0f}s elapsed')

    # ── Aggregate results ──
    all_tanimotos = np.array(all_tanimotos)  # (N_val, 3)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for k in range(3):
        vals = all_tanimotos[:, k]
        print(f"\nTop-{k+1} Tanimoto:")
        print(f"  Mean:   {vals.mean():.4f}")
        print(f"  Median: {np.median(vals):.4f}")
        print(f"  Std:    {vals.std():.4f}")
        print(f"  > 0.4:  {(vals > 0.4).mean() * 100:.1f}%")
        print(f"  > 0.7:  {(vals > 0.7).mean() * 100:.1f}%")
        print(f"  > 0.9:  {(vals > 0.9).mean() * 100:.1f}%")

    print(f"\nTop-3 Average Tanimoto:")
    avg_tanimoto = all_tanimotos.mean(axis=1)
    print(f"  Mean:   {avg_tanimoto.mean():.4f}")
    print(f"  Median: {np.median(avg_tanimoto):.4f}")

    print(f"\nWorst cases (lowest Top-1 Tanimoto):")
    worst_idx = np.argsort(all_tanimotos[:, 0])[:5]
    for i in worst_idx:
        print(f"  Query:   {val_smiles[i]}")
        for k in range(3):
            print(f"    Neighbor {k+1}: Tanimoto={all_tanimotos[i, k]:.4f}")

    print(f"\nBest cases (highest Top-1 Tanimoto):")
    best_idx = np.argsort(all_tanimotos[:, 0])[-5:]
    for i in best_idx:
        print(f"  Query:   {val_smiles[i]}")
        for k in range(3):
            print(f"    Neighbor {k+1}: Tanimoto={all_tanimotos[i, k]:.4f}")

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print("\nVerdict:")
    if avg_tanimoto.mean() > 0.4:
        print("  ✅ Mean Top-3 Tanimoto > 0.4: RAG foundation is solid.")
    elif avg_tanimoto.mean() > 0.25:
        print("  ⚠️  Mean Top-3 Tanimoto between 0.25-0.4: RAG may help modestly.")
    else:
        print("  ❌ Mean Top-3 Tanimoto < 0.25: RAG foundation is weak.")
        print("     Consider contrastive learning to align MS emb with molecular fingerprint.")


if __name__ == '__main__':
    main()
