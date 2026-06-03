"""Pre-cache full_embedding subset to local HDF5 for training.
Reads 40k sorted slices from the FUSE-hosted HDF5 (once, slowly),
writes to local disk (~9.8GB) so subsequent reads are fast.
"""
import h5py
import numpy as np
import os
import time

SRC = '/root/datasets/pairs_with_embs.hdf5'
DST = '/tmp/fullseq_subset_40000.h5'
N = 40000
SEED = 42

def main():
    t0 = time.time()

    with h5py.File(SRC, 'r') as src:
        split = src['split'][:]
        train_mask = split == 0
        orig_indices_arr = np.where(train_mask)[0]

        # Read all SMILES for grouping
        print('Reading SMILES from FUSE...', flush=True)
        all_smiles = [s.decode() if isinstance(s, bytes) else s
                      for s in src['smiles'][:]]
        smiles = [s for s, m in zip(all_smiles, train_mask) if m]
        del all_smiles

        print(f'Grouping {len(smiles)} unique molecules...', flush=True)
        mol_to_indices = {}
        for i, smi in enumerate(smiles):
            mol_to_indices.setdefault(smi, []).append(i)
        unique_smiles = list(mol_to_indices.keys())
        n_unique = len(unique_smiles)
        n_sample = min(N, n_unique)

        rng = np.random.RandomState(SEED)
        chosen_smiles = rng.choice(unique_smiles, n_sample, replace=False)

        sample_indices = []
        kept_smiles = []
        for smi in chosen_smiles:
            idx = rng.choice(mol_to_indices[smi])
            sample_indices.append(idx)
            kept_smiles.append(smi)

        # Map to original HDF5 row indices and sort for sequential reading
        row_idxs = np.sort(orig_indices_arr[sample_indices])
        del smiles, unique_smiles, chosen_smiles, mol_to_indices

        print(f'Reading {n_sample} full_embedding slices (sorted)...', flush=True)
        ds = src['full_embedding']
        shape = ds.shape[1:]  # (60, 1024)
        sorted_embs = np.empty((n_sample, *shape), dtype=np.float32)
        for i, row_idx in enumerate(row_idxs):
            sorted_embs[i] = ds[row_idx]
            if (i + 1) % 5000 == 0:
                print(f'  {i+1}/{n_sample} ({time.time()-t0:.0f}s)', flush=True)

    # Reorder back to match sample_indices order
    print('Reordering...', flush=True)
    sort_order = np.argsort(orig_indices_arr[sample_indices])
    unsort_order = np.argsort(sort_order)
    embs = sorted_embs[unsort_order]

    # Write to local HDF5
    print(f'Writing to {DST}...', flush=True)
    if os.path.exists(DST):
        os.remove(DST)
    with h5py.File(DST, 'w') as dst:
        dst.create_dataset('full_embedding', data=embs)
        dst.create_dataset('smiles', data=[s.encode() for s in kept_smiles],
                          dtype=h5py.string_dtype())

    elapsed = time.time() - t0
    print(f'Done! {DST} ({embs.shape}, {os.path.getsize(DST)/1e9:.1f}GB) in {elapsed:.0f}s')

if __name__ == '__main__':
    main()
