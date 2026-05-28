"""
Precompute MACCS keys for all spectra and save to a new field in pairs_ready.hdf5.
Run once before training:
    python dreams/training/precompute_maccs.py

This eliminates the RDKit bottleneck in the training DataLoader.
"""

import sys, os, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/root/DreaMS')

import h5py
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys

HDF5_PATH = '/root/datasets/pairs_ready.hdf5'
MACCS_NBITS = 167

print(f'Loading {HDF5_PATH} ...')
with h5py.File(HDF5_PATH, 'r') as f:
    smiles = [s.decode() if isinstance(s, bytes) else s for s in f['smiles'][:]]
    n_total = len(smiles)
print(f'Total: {n_total}')

# Compute MACCS for all SMILES
maccs = np.zeros((n_total, MACCS_NBITS), dtype='float32')
for i, smi in enumerate(tqdm(smiles, desc='MACCS')):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fp = MACCSkeys.GenMACCSKeys(mol)
        maccs[i] = [fp.GetBit(j) for j in range(MACCS_NBITS)]

# Write back (append mode)
print(f'Writing MACCS to {HDF5_PATH} ...')
with h5py.File(HDF5_PATH, 'a') as f:
    if 'maccs' in f:
        del f['maccs']
        print('  Replaced existing maccs field')
    f.create_dataset('maccs', data=maccs, dtype='float32',
                     compression='gzip', compression_opts=6)

print(f'Done! MACCS shape: {maccs.shape}, size: {maccs.nbytes / 1e6:.0f} MB')
