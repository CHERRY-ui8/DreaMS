"""
prepare_pairs.py — 生成 (MS_embedding, SMILES) 二元组数据集

流程:
  1. 从 dreams_ready.hdf5 读取所有谱图
  2. SMILES canonicalization + 去盐 + 可选去手性
  3. 每分子最多保留 MAX_SPECTRA_PER_MOL 张谱图
  4. 按分子 (SMILES) 做 Group Split → train / val / test
  5. 输出清洗后的中间 HDF5: pairs_ready.hdf5
  6. [可选] 运行 DreaMS 抽取 embedding

用法:
  python3 prepare_pairs.py                      # 只做 step 1-5
  python3 prepare_pairs.py --extract-embeddings  # 包括 step 6
"""

import sys, os, warnings, argparse, random
from pathlib import Path
warnings.filterwarnings('ignore')

import h5py
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from collections import defaultdict, Counter

# ─── Config ─────────────────────────────────────
HDF5_IN = Path('/root/datasets/dreams_ready.hdf5')
HDF5_OUT = Path('/root/datasets/pairs_ready.hdf5')
EMBEDDING_HDF5 = Path('/root/datasets/pairs_with_embs.hdf5')

# ADDUCT mass offset table (M + adduct_mass - electron_mass)
ADDUCT_MASS_OFFSET = {
    '[M+H]+': 1.007276,
    '[M-H]-': -1.007276,
    '[M+Na]+': 22.989218,
    '[M+K]+': 38.963158,
    '[M+NH4]+': 18.033823,
    '[M+CH3OH+H]+': 33.033489,
    '[M+ACN+H]+': 42.033823,
    '[M+ACN+Na]+': 63.032823,
    '[M+2H]2+': 1.007276 * 2,
    '[M+3H]3+': 1.007276 * 3,
    '[M-H2O-H]-': -19.017841,
    '[M+Cl]-': 34.968853,
    '[M+Br]-': 78.918338,
    '[M+CH3COO]-': 59.013851,
    '[M+HCOO]-': 45.002008,
    '[2M+H]+': -0.001,  # will be handled separately
    '[2M-H]-': -0.001,
    '[M+ACETATE]-': 59.013851,
    '[M+H-H2O]+': 1.007276 - 18.010565,
    '[2M+Na]+': -0.001,
}

# ─── Args ───────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--max-per-mol', type=int, default=10,
                    help='Max spectra per molecule (cap for imbalance)')
parser.add_argument('--val-split', type=float, default=0.05,
                    help='Fraction of molecules for validation')
parser.add_argument('--test-split', type=float, default=0.1,
                    help='Fraction of molecules for test')
parser.add_argument('--mass-tol', type=float, default=0.5,
                    help='Precursor m/z tolerance for mass matching (Da)')
parser.add_argument('--no-stereo', action='store_true', default=True,
                    help='Remove stereochemistry from SMILES')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--extract-embeddings', action='store_true',
                    help='Run DreaMS embedding extraction after cleaning')
args = parser.parse_args()


def canonicalize_smiles(smiles: str, no_stereo: bool = True) -> str:
    """Canonicalize SMILES: desalt, canonicalize, optionally strip stereochemistry."""
    if not smiles or smiles in ('', 'N/A', 'None'):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Desalt: keep largest fragment
    frags = Chem.GetMolFrags(mol, asMols=True)
    mol = max(frags, key=lambda m: m.GetNumAtoms())
    
    # Remove stereochemistry
    if no_stereo:
        Chem.RemoveStereochemistry(mol)
    
    # Canonicalize
    return Chem.MolToSmiles(mol)


def exact_mass_from_smiles(smiles: str) -> float:
    """Compute exact mass from canonical SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Descriptors.ExactMolWt(mol)


def check_mass_match(precursor_mz: float, exact_mass: float, adduct: str, tol: float = 0.5) -> bool:
    """Check if precursor_mz matches exact_mass + adduct offset within tolerance."""
    if np.isnan(precursor_mz) or precursor_mz <= 0:
        return True  # skip check if no valid precursor_mz
    if exact_mass is None:
        return True
    
    offset = ADDUCT_MASS_OFFSET.get(adduct, None)
    if offset is None:
        # Try to guess from adduct string
        if '+H]+' in adduct or '+H]' in adduct:
            offset = 1.007276
        elif '-H]-' in adduct or '-H]' in adduct:
            offset = -1.007276
        elif '+Na]+' in adduct:
            offset = 22.989218
        elif '+K]+' in adduct:
            offset = 38.963158
        elif '+NH4]+' in adduct:
            offset = 18.033823
        elif '+Cl]-' in adduct:
            offset = 34.968853
        elif '+CH3COO]-' in adduct or '+ACETATE]-' in adduct:
            offset = 59.013851
        elif '+HCOO]-' in adduct:
            offset = 45.002008
        elif '+Br]-' in adduct:
            offset = 78.918338
        else:
            return True  # unknown adduct, skip check
    
    expected_mz = exact_mass + offset
    diff = abs(precursor_mz - expected_mz)
    return diff <= tol


# ═══════════════════════════════════════════════════
# Step 1-2: Load HDF5, canonicalize SMILES, mass check
# ═══════════════════════════════════════════════════
print('=' * 60)
print('Step 1: Loading and canonicalizing SMILES')
print('=' * 60)

with h5py.File(HDF5_IN, 'r') as f:
    n_total = len(f['smiles'])
    smiles_raw = f['smiles'][:]
    precursor_mz = f['precursor_mz'][:]
    adduct = f['adduct'][:]
    charge = f['charge'][:]
    ce = f['collision_energy'][:]
    spectrum = f['spectrum'][:]

print(f'Total spectra loaded: {n_total}')

# Canonicalize
canonical_map = {}  # original -> canonical
passed_mass = 0
failed_mass = 0
no_mass_info = 0
records = []

for i in tqdm(range(n_total), desc='Canonicalizing'):
    sm = smiles_raw[i].decode() if isinstance(smiles_raw[i], bytes) else smiles_raw[i]
    can = canonicalize_smiles(sm, no_stereo=args.no_stereo)
    if can is None:
        continue
    
    # Mass matching
    exact_mass = exact_mass_from_smiles(can)
    ad_str = adduct[i].decode() if isinstance(adduct[i], bytes) else adduct[i]
    pmz = float(precursor_mz[i])
    
    if np.isnan(pmz) or pmz <= 0:
        no_mass_info += 1
        mass_ok = True
    elif check_mass_match(pmz, exact_mass, ad_str, tol=args.mass_tol):
        passed_mass += 1
        mass_ok = True
    else:
        failed_mass += 1
        mass_ok = False
    
    if not mass_ok:
        continue
    
    records.append({
        'idx': i,
        'smiles': can,
        'precursor_mz': pmz,
        'adduct': ad_str,
        'charge': int(charge[i]),
        'ce': int(ce[i]),
        'spectrum': spectrum[i],
    })

print(f'\nMass matching: {passed_mass} passed, {failed_mass} failed, {no_mass_info} no mass info')
print(f'Records after cleaning: {len(records)}')

# ═══════════════════════════════════════════════════
# Step 3: Cap spectra per molecule
# ═══════════════════════════════════════════════════
print('\n' + '=' * 60)
print(f'Step 2: Capping at {args.max_per_mol} spectra per molecule')
print('=' * 60)

# Group by SMILES
mol_groups = defaultdict(list)
for rec in records:
    mol_groups[rec['smiles']].append(rec)

total_before_cap = len(records)
total_after_cap = 0
capped_records = []

for sm, group in tqdm(mol_groups.items(), desc='Capping'):
    if len(group) <= args.max_per_mol:
        capped_records.extend(group)
        total_after_cap += len(group)
    else:
        # Subsample: try to pick diverse CE/adduct combinations
        # Prioritize variety: pick spectra with different CE values first
        random.Random(args.seed).shuffle(group)
        
        # Strategy: pick diverse CEs
        sorted_group = sorted(group, key=lambda x: x['ce'])
        step = len(sorted_group) / args.max_per_mol
        selected = [sorted_group[int(i * step)] for i in range(args.max_per_mol)]
        capped_records.extend(selected)
        total_after_cap += len(selected)

print(f'Before cap: {total_before_cap} spectra, {len(mol_groups)} molecules')
print(f'After cap:  {total_after_cap} spectra')

# ═══════════════════════════════════════════════════
# Step 4: Group split by molecule (SMILES)
# ═══════════════════════════════════════════════════
print('\n' + '=' * 60)
print('Step 3: Group split by molecule (zero-shot)')
print('=' * 60)

# Get unique SMILES
unique_smiles = list(set(rec['smiles'] for rec in capped_records))
random.Random(args.seed).shuffle(unique_smiles)

n_val = int(len(unique_smiles) * args.val_split)
n_test = int(len(unique_smiles) * args.test_split)
n_train = len(unique_smiles) - n_val - n_test

val_smiles = set(unique_smiles[:n_val])
test_smiles = set(unique_smiles[n_val:n_val + n_test])
train_smiles = set(unique_smiles[n_val + n_test:])

print(f'Train molecules: {len(train_smiles)}')
print(f'Val   molecules: {len(val_smiles)}')
print(f'Test  molecules: {len(test_smiles)}')

def assign_split(sm):
    if sm in train_smiles:
        return 0  # train
    elif sm in val_smiles:
        return 1  # val
    else:
        return 2  # test

split_counts = [0, 0, 0]
for rec in capped_records:
    rec['split'] = assign_split(rec['smiles'])
    split_counts[rec['split']] += 1

print(f'Train spectra: {split_counts[0]}')
print(f'Val   spectra: {split_counts[1]}')
print(f'Test  spectra: {split_counts[2]}')

# ═══════════════════════════════════════════════════
# Step 5: Write intermediate HDF5
# ═══════════════════════════════════════════════════
print('\n' + '=' * 60)
print(f'Step 4: Writing {HDF5_OUT}')
print('=' * 60)

n_out = len(capped_records)
max_peaks = capped_records[0]['spectrum'].shape[1]  # should be 60

with h5py.File(HDF5_OUT, 'w') as f:
    # Store index mapping for later embedding extraction
    ds_smiles = f.create_dataset('smiles', (n_out,), dtype=h5py.string_dtype())
    ds_smiles_orig = f.create_dataset('smiles_original', (n_out,), dtype=h5py.string_dtype())
    ds_adduct = f.create_dataset('adduct', (n_out,), dtype=h5py.string_dtype())
    ds_split = f.create_dataset('split', (n_out,), dtype='int8')
    ds_ce = f.create_dataset('collision_energy', (n_out,), dtype='int16')
    ds_charge = f.create_dataset('charge', (n_out,), dtype='int8')
    ds_precursor_mz = f.create_dataset('precursor_mz', (n_out,), dtype='float32')
    ds_spectrum = f.create_dataset('spectrum', (n_out, 2, max_peaks), dtype='float32')
    ds_orig_idx = f.create_dataset('original_index', (n_out,), dtype='int32')
    
    for i, rec in enumerate(tqdm(capped_records, desc='Writing')):
        ds_smiles[i] = rec['smiles']
        ds_smiles_orig[i] = smiles_raw[rec['idx']].decode() if isinstance(smiles_raw[rec['idx']], bytes) else smiles_raw[rec['idx']]
        ds_adduct[i] = rec['adduct']
        ds_split[i] = rec['split']
        ds_ce[i] = rec['ce']
        ds_charge[i] = rec['charge']
        ds_precursor_mz[i] = rec['precursor_mz']
        ds_spectrum[i] = rec['spectrum']
        ds_orig_idx[i] = rec['idx']
    
    # Store split metadata
    f.attrs['train_molecules'] = len(train_smiles)
    f.attrs['val_molecules'] = len(val_smiles)
    f.attrs['test_molecules'] = len(test_smiles)
    f.attrs['train_spectra'] = split_counts[0]
    f.attrs['val_spectra'] = split_counts[1]
    f.attrs['test_spectra'] = split_counts[2]
    f.attrs['max_spectra_per_mol'] = args.max_per_mol
    f.attrs['no_stereo'] = args.no_stereo

print(f'Written to {HDF5_OUT}')
print(f'  Spectra: {n_out} (from {n_total} original, {total_before_cap} after cleaning)')
print(f'  Unique molecules: {len(unique_smiles)}')
print(f'  Mass tolerance: {args.mass_tol} Da')
print(f'  Max spectra/mol: {args.max_per_mol}')
print(f'  Stereochemistry: {"removed" if args.no_stereo else "kept"}')

# ═══════════════════════════════════════════════════
# Step 6 (optional): DreaMS embedding extraction
# ═══════════════════════════════════════════════════
if args.extract_embeddings:
    print('\n' + '=' * 60)
    print('Step 5: Extracting DreaMS embeddings')
    print('=' * 60)
    
    # ── DreaMS imports (need to be after CUDA init) ──
    import torch
    if torch.cuda.is_available():
        _ = torch.zeros(1, device='cuda')
        del _
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    import sys as _sys
    import types as _types
    import dreams.utils.data as du
    import dreams.utils.dformats as dformats
    import dreams.utils.spectra as su
    import dreams.models.dreams.dreams as dm
    import dreams.models.dreams.layers as dl
    import dreams.models.layers.fourier_features as ff
    import dreams.models.layers.feed_forward as fw
    for ns in ['msml','msml.models','msml.models.dreams','msml.models.layers','msml.utils']:
        _sys.modules[ns] = _types.ModuleType(ns)
    _sys.modules['msml.models.dreams.dreams'] = dm
    _sys.modules['msml.models.dreams.layers'] = dl
    _sys.modules['msml.models.layers.fourier_features'] = ff
    _sys.modules['msml.models.layers.feed_forward'] = fw
    _sys.modules['msml.utils.data'] = du
    _sys.modules['msml.utils.dformats'] = dformats
    _sys.modules['msml.utils.spectra'] = su
    
    from argparse import Namespace
    from dreams.utils.data import SpectrumPreprocessor, MSData
    from dreams.utils.dformats import DataFormatA
    from dreams.models.dreams.dreams import DreaMS
    from torch.utils.data import DataLoader, TensorDataset
    
    # Load DreaMS backbone
    CKPT = 'dreams/models/pretrained/ssl_model.ckpt'
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
    clean['enable_cond_tokens'] = False  # NO cond tokens
    
    model = DreaMS(Namespace(**clean), spec_preproc)
    load_sd = {k: v for k, v in sd.items() if not any(x in k for x in ['ro_out', 'mz_masking'])}
    model.load_state_dict(load_sd, strict=False)
    model = model.to(device)
    model.eval()
    print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} params')
    
    # Read back the cleaned HDF5
    with h5py.File(HDF5_OUT, 'r') as f:
        spectra_data = f['spectrum'][:]
        smiles_data = f['smiles'][:]
        split_data = f['split'][:]
    
    # Process in batches
    batch_size = 256
    n_spectra = len(spectra_data)
    embeddings = np.zeros((n_spectra, 1024), dtype='float32')
    
    print(f'Extracting embeddings for {n_spectra} spectra...')
    with torch.no_grad():
        for start in tqdm(range(0, n_spectra, batch_size)):
            end = min(start + batch_size, n_spectra)
            batch_spectra = spectra_data[start:end]
            
            # Convert to preprocessed format: peaks need to be [mz, intensity]
            # spectrum is stored as (2, 60) = [mz_values, intensity_values]
            peaks = torch.from_numpy(batch_spectra).float().to(device)  # (B, 2, 60)
            
            # DreaMS forward expects (B, n_peaks, 2) where last dim is (mz, intensity)
            # Our data is (2, 60), so permute
            peaks = peaks.permute(0, 2, 1)  # (B, 60, 2)
            
            # Apply spectrum preprocessor
            # The model's forward does this internally, but we can also do it here
            # Actually, DreaMS.forward takes raw peaks and applies preprocessor
            # But the forward expects the original input format
            embs = model(peaks, charge=None)
            
            if embs.dim() == 3:
                embs = embs[:, 0, :]  # (B, 1024) - prending token
            
            embeddings[start:end] = embs.cpu().numpy()
    
    # Write embedding HDF5
    print(f'\nWriting {EMBEDDING_HDF5}...')
    with h5py.File(EMBEDDING_HDF5, 'w') as f:
        f.create_dataset('embedding', data=embeddings, dtype='float32')
        f.create_dataset('smiles', data=smiles_data, dtype=h5py.string_dtype())
        f.create_dataset('split', data=split_data, dtype='int8')
        
        # Copy additional fields
        with h5py.File(HDF5_OUT, 'r') as fin:
            for key in ['adduct', 'collision_energy', 'charge', 'precursor_mz']:
                fin.copy(key, f, key)
        
        f.attrs.update(HDF5_OUT.with_name('pairs_ready.hdf5'))
        # Copy attrs from source
        with h5py.File(HDF5_OUT, 'r') as fin:
            for k, v in fin.attrs.items():
                f.attrs[k] = v
    
    print(f'Written to {EMBEDDING_HDF5}')
    print(f'  Embedding shape: {embeddings.shape}')
    print(f'  Memory: {embeddings.nbytes / 1e6:.0f} MB')

print('\nDone!')
