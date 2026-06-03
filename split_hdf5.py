"""Split the large full_embedding HDF5 into per-split files — memory-efficient chunked version.

Reads the source HDF5 in chunks of CHUNK_SIZE rows, routes each row to
its split output file, and writes incrementally. Peak memory ~CHUNK_SIZE * 60 * 1024 * 4 bytes.
"""

import h5py
import numpy as np
from pathlib import Path

H5_IN = Path('/root/datasets/pairs_with_embs.hdf5')
CHUNK_SIZE = 5000  # ~5000 * 60 * 1024 * 4 = ~1.2 GB per chunk

split_names = {0: 'train', 1: 'val', 2: 'test'}

print(f'Reading split + smiles + maccs indices from {H5_IN}...')
with h5py.File(H5_IN, 'r') as f:
    split_arr = f['split'][:]    # (N,) int8 — tiny
    smiles_ds = f['smiles'][:]   # (N,) strings
    has_maccs = 'maccs' in f
    N = len(split_arr)

print(f'Total: {N} spectra, has_maccs={has_maccs}')

# Count per split
for sv, sn in split_names.items():
    print(f'  {sn}: {(split_arr == sv).sum()}')

# Open output files and create datasets (empty, extensible)
print('\nOpening output files...')
out_files = {}
out_ds = {}
out_smiles = {}
out_maccs = {}

for sv, sn in split_names.items():
    path = H5_IN.parent / f'pairs_with_embs_{sn}.hdf5'
    # Write mode — create new
    f_out = h5py.File(path, 'w')
    out_files[sv] = f_out

    # Create extensible datasets
    # full_embedding: (N_split, 60, 1024) with row-oriented chunks
    out_ds[sv] = f_out.create_dataset(
        'full_embedding',
        shape=(0, 60, 1024),
        maxshape=(None, 60, 1024),
        chunks=(1, 60, 1024),
        compression='gzip', compression_opts=1,
        dtype='float32',
    )
    # SMILES
    out_smiles[sv] = f_out.create_dataset(
        'smiles',
        shape=(0,),
        maxshape=(None,),
        dtype=h5py.string_dtype(),
    )
    # MACCS (if source has it)
    if has_maccs:
        out_maccs[sv] = f_out.create_dataset(
            'maccs',
            shape=(0, 167),
            maxshape=(None, 167),
            dtype='int8',
        )

# Write counters per split
counters = {sv: 0 for sv in split_names}

# Chunked read + route
print(f'\nProcessing in chunks of {CHUNK_SIZE}...')
with h5py.File(H5_IN, 'r') as f_in:
    full_emb_ds = f_in['full_embedding']
    maccs_ds = f_in['maccs'] if has_maccs else None

    for start in range(0, N, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, N)
        print(f'  chunk [{start}:{end}] ({end-start} rows)...', end=' ', flush=True)

        # Read chunk
        emb_chunk = full_emb_ds[start:end]      # (B, 60, 1024)
        smiles_chunk = smiles_ds[start:end]
        maccs_chunk = maccs_ds[start:end] if has_maccs else None
        split_chunk = split_arr[start:end]

        # Route each row to its split file
        for i in range(end - start):
            sv = int(split_chunk[i])  # 0=train, 1=val, 2=test
            if sv not in split_names:
                continue
            cnt = counters[sv]

            # Append to full_embedding
            ds = out_ds[sv]
            ds.resize(cnt + 1, axis=0)
            ds[cnt] = emb_chunk[i]

            # Append to smiles
            ss = out_smiles[sv]
            ss.resize(cnt + 1, axis=0)
            ss[cnt] = smiles_chunk[i]

            # Append to maccs if available
            if has_maccs and sv in out_maccs:
                ms = out_maccs[sv]
                ms.resize(cnt + 1, axis=0)
                ms[cnt] = maccs_chunk[i]

            counters[sv] = cnt + 1

        print(f'counters: {dict(counters)}')

# Also write pooled embedding for backward compat
print('\nWriting pooled embedding for backward compatibility...')
for sv, sn in split_names.items():
    f_out = out_files[sv]
    n = counters[sv]
    if n == 0:
        continue
    pooled = f_out['full_embedding'][:, 0, :]  # (N, 1024)
    f_out.create_dataset('embedding', data=pooled, dtype='float32')

# Close all files
for sv, f_out in out_files.items():
    f_out.close()

# Print summary
print('\n=== Done ===')
for sv, sn in split_names.items():
    path = H5_IN.parent / f'pairs_with_embs_{sn}.hdf5'
    size_mb = path.stat().st_size / 1e6 if path.exists() else 0
    print(f'  {sn}: {counters[sv]} spectra, {size_mb:.0f} MB')
