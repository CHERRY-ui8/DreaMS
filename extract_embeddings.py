"""
extract_embeddings.py — 用 DreaMS (无 cond token) 抽取 MS embedding

输入: /root/datasets/pairs_ready.hdf5 (spectra + SMILES + split)
输出: /root/datasets/pairs_with_embs.hdf5 (同上 + embedding 列)
"""

import sys, types, os, warnings, time
from pathlib import Path
warnings.filterwarnings('ignore')

# ── CUDA init before TensorFlow ──
import torch
if torch.cuda.is_available():
    _ = torch.zeros(1, device='cuda')
    del _
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
assert device == 'cuda', 'CUDA not available!'

# ── msml mock ──
import dreams.utils.data as du; import dreams.utils.dformats as dformats
import dreams.utils.spectra as su; import dreams.models.dreams.dreams as dm
import dreams.models.dreams.layers as dl
import dreams.models.layers.fourier_features as ff
import dreams.models.layers.feed_forward as fw
for ns in ['msml','msml.models','msml.models.dreams','msml.models.layers','msml.utils']:
    sys.modules[ns] = types.ModuleType(ns)
sys.modules['msml.models.dreams.dreams'] = dm
sys.modules['msml.models.dreams.layers'] = dl
sys.modules['msml.models.layers.fourier_features'] = ff
sys.modules['msml.models.layers.feed_forward'] = fw
sys.modules['msml.utils.data'] = du; sys.modules['msml.utils.dformats'] = dformats
sys.modules['msml.utils.spectra'] = su

import numpy as np
import h5py
from argparse import Namespace
from tqdm import tqdm
from dreams.utils.data import SpectrumPreprocessor, MSData
from dreams.utils.dformats import DataFormatA
from dreams.models.dreams.dreams import DreaMS
from torch.utils.data import DataLoader, TensorDataset

# ─── Config ─────────────────────────────────────
HDF5_IN = Path('/root/datasets/pairs_ready.hdf5')
HDF5_OUT = Path('/root/datasets/pairs_with_embs.hdf5')
CKPT = 'dreams/models/pretrained/ssl_model.ckpt'
BATCH_SIZE = 64

# ─── Output mode ─────────────────────────────────
# 'pooled' — (B, 1024)  [CLS] token only  (original behavior)
# 'full'   — (B, 60, 1024)  all peak positions
MODE = 'full'

# ─── Load DreaMS ─────────────────────────────────
print('Loading DreaMS backbone...')
raw = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = raw['state_dict']
hp = raw['hyper_parameters']['args']
if hasattr(hp, '__dict__'): hp = vars(hp)

dformat = DataFormatA()
spec_preproc = SpectrumPreprocessor(dformat=dformat, n_highest_peaks=60)

clean = {'dformat': dformat, 'no_transformer_bias': True}
for k, v in hp.items():
    if isinstance(v, Path): clean[k] = str(v)
    else: clean[k] = v
clean['enable_cond_tokens'] = False

model = DreaMS(Namespace(**clean), spec_preproc)
load_sd = {k: v for k, v in sd.items() if not any(x in k for x in ['ro_out', 'mz_masking'])}
model.load_state_dict(load_sd, strict=False)
model = model.to(device)
model.eval()
print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')

# ─── Load HDF5 ───────────────────────────────────
print('Loading spectra...')
with h5py.File(HDF5_IN, 'r') as f:
    spectra = f['spectrum'][:]       # (N, 2, 60)
    smiles = f['smiles'][:]
    split = f['split'][:]
    adduct = f['adduct'][:]
    ce = f['collision_energy'][:]
    charge = f['charge'][:]
    pmz = f['precursor_mz'][:]

n_total = len(spectra)
print(f'Spectra to process: {n_total}')

# ─── Extract embeddings ─────────────────────────
print(f'Mode: {MODE} ({"(60, 1024) full sequence" if MODE == "full" else "(1024,) pooled [CLS]"})')
if MODE == 'full':
    embeddings = np.zeros((n_total, 60, 1024), dtype='float32')
else:
    embeddings = np.zeros((n_total, 1024), dtype='float32')

with torch.no_grad():
    for start in tqdm(range(0, n_total, BATCH_SIZE), desc='Embedding'):
        end = min(start + BATCH_SIZE, n_total)
        batch_spectra = spectra[start:end]

        # Convert to tensor: (B, 2, 60) -> (B, 60, 2) where last dim is [mz, intensity]
        peaks = torch.from_numpy(batch_spectra).float().to(device)
        peaks = peaks.permute(0, 2, 1)  # (B, 60, 2)

        # DreaMS forward — it applies SpectrumPreprocessor internally
        embs = model(peaks, charge=None)

        if MODE == 'full':
            # Save all 60 positions: (B, 60, 1024)
            embeddings[start:end] = embs.cpu().numpy()
        else:
            # Original behavior: only [CLS] token
            if embs.dim() == 3:
                embs = embs[:, 0, :]  # (B, 1024)
            embeddings[start:end] = embs.cpu().numpy()

print(f'Embeddings shape: {embeddings.shape}')
print(f'Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]')

# ─── Write output ─────────────────────────────────
print(f'Writing {HDF5_OUT}...')
with h5py.File(HDF5_OUT, 'w') as f:
    if MODE == 'full':
        dataset_name = 'full_embedding'
    else:
        dataset_name = 'embedding'
    f.create_dataset(dataset_name, data=embeddings, dtype='float32',
                     compression='gzip', compression_opts=6)

    # Always save pooled embedding too for backward compatibility
    if MODE == 'full':
        pooled = embeddings[:, 0, :]  # (N, 1024) — [CLS] token
        f.create_dataset('embedding', data=pooled, dtype='float32',
                         compression='gzip', compression_opts=6)

    f.create_dataset('smiles', data=smiles, dtype=h5py.string_dtype())
    f.create_dataset('split', data=split, dtype='int8')
    f.create_dataset('adduct', data=adduct, dtype=h5py.string_dtype())
    f.create_dataset('collision_energy', data=ce, dtype='int16')
    f.create_dataset('charge', data=charge, dtype='int8')
    f.create_dataset('precursor_mz', data=pmz, dtype='float32')

    # Copy precomputed MACCS from pairs_ready.hdf5
    with h5py.File(HDF5_IN, 'r') as fin:
        if 'maccs' in fin:
            f.create_dataset('maccs', data=fin['maccs'][:], dtype='int8')

    # Copy attrs
    with h5py.File(HDF5_IN, 'r') as fin:
        for k, v in fin.attrs.items():
            f.attrs[k] = v

size_mb = os.path.getsize(HDF5_OUT) / 1e6
print(f'Done! {HDF5_OUT} ({size_mb:.0f} MB)')
emb_name = 'full_embedding' if MODE == 'full' else 'embedding'
print(f'  Dataset: {emb_name} {embeddings.shape}')
print(f'  Backward compat: embedding (1024,) also saved' if MODE == 'full' else '')
print(f'  Memory: {embeddings.nbytes / 1e6:.0f} MB (float32)')
