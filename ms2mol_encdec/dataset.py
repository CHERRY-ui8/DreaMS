"""Dataset for T5-based MS → SMILES training.

T5 handles SMILES natively (SentencePiece, 0 UNK). No SELFIES conversion needed.
Tokenization done in collate_fn for efficiency (batch tokenization with padding).
"""

import h5py
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import MACCSkeys


class MSSpectrumSmilesT5Dataset(Dataset):
    """Dataset of (MS embedding, SMILES) pairs for T5 training.

    SMILES strings are tokenized in collate_fn using T5's batch tokenizer,
    which handles padding and label creation automatically.
    """

    def __init__(
        self,
        hdf5_path: str = '/root/datasets/pairs_with_embs.hdf5',
        split: str = 'train',
    ):
        print(f'[Dataset] Loading {split} from {hdf5_path}')
        with h5py.File(hdf5_path, 'r') as f:
            split_data = f['split'][:]
            mask = split_data == {'train': 0, 'val': 1, 'test': 2}[split]

            all_embs = f['embedding'][:]
            self.embeddings = all_embs[mask]
            print(f'  Embeddings: {self.embeddings.shape}')

            all_smiles = f['smiles'][:]
            self.smiles = [s.decode() if isinstance(s, bytes) else s
                           for s, m in zip(all_smiles, mask) if m]

        print(f'  Spectra: {len(self)}')

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return {
            'ms_emb': torch.from_numpy(self.embeddings[idx]).float(),
            'smiles': self.smiles[idx],  # raw SMILES string, tokenized in collate
            'maccs': smi_to_maccs(self.smiles[idx]),  # (167,) binary tensor
        }


def smi_to_maccs(smiles: str) -> torch.Tensor:
    """Convert a SMILES string to a 167-bit MACCS key fingerprint tensor.

    Returns:
        (MACCS_NBITS,) binary float tensor. Returns zeros if RDKit can't parse.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(167, dtype=torch.float)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return torch.tensor([fp.GetBit(i) for i in range(167)], dtype=torch.float)


class MSSpectrumSmilesFullSeqDataset(Dataset):
    """Dataset of (full MS sequence, SMILES, precomputed MACCS) for T5 training.

    Loads the full (60, 1024) DreaMS peak sequence instead of pooled (1024,).
    Requires extract_embeddings.py run with MODE='full'.
    MACCS is loaded from HDF5 (precomputed by precompute_maccs.py).
    """

    def __init__(
        self,
        hdf5_path: str = '/root/datasets/pairs_with_embs.hdf5',
        split: str = 'train',
    ):
        print(f'[Dataset] Loading {split} from {hdf5_path} (lazy: index-only)')
        self.hdf5_path = hdf5_path
        self.split = split
        with h5py.File(hdf5_path, 'r') as f:
            split_data = f['split'][:]
            split_val = {'train': 0, 'val': 1, 'test': 2}[split]
            self.indices = np.where(split_data == split_val)[0]
            print(f'  Found {len(self.indices)} spectra of split={split}')

            all_smiles = f['smiles'][:]
            self.smiles = [s.decode() if isinstance(s, bytes) else s
                           for s in all_smiles[self.indices]]

            if 'maccs' in f:
                # Precomputed MACCS loaded eagerly (167-d, small)
                self.maccs = f['maccs'][self.indices]
                print(f'  Precomputed MACCS: {self.maccs.shape}')
            else:
                self.maccs = None
                print(f'  No precomputed MACCS — will compute on-the-fly')

        # Open persistent HDF5 handle for lazy embedding loading
        self._h5_file = h5py.File(hdf5_path, 'r')
        self._full_emb_ds = self._h5_file['full_embedding']

        print(f'  Spectra: {len(self)}')

    def __del__(self):
        if hasattr(self, '_h5_file'):
            self._h5_file.close()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = {
            'ms_emb': torch.from_numpy(self._full_emb_ds[real_idx]).float(),  # (60, 1024)
            'smiles': self.smiles[idx],
        }
        if self.maccs is not None:
            item['maccs'] = torch.from_numpy(self.maccs[idx]).float()
        else:
            item['maccs'] = smi_to_maccs(self.smiles[idx])
        return item


class NSubsetDataset(Dataset):
    """Load N random samples from the HDF5 training split for overfitting tests."""

    def __init__(self, hdf5_path='/root/datasets/pairs_with_embs.hdf5', n=100, seed=42):
        with h5py.File(hdf5_path, 'r') as f:
            all_split = f['split'][:]
            mask = all_split == 0  # train split
            all_embs = f['embedding'][:][mask]
            all_smiles = [s.decode() if isinstance(s, bytes) else s
                          for s, m in zip(f['smiles'][:], mask) if m]

        total = len(all_smiles)
        rng = np.random.RandomState(seed)
        indices = rng.choice(total, min(n, total), replace=False)
        indices.sort()

        self.smiles = [all_smiles[i] for i in indices]
        self.embeddings = all_embs[indices]

        lengths = [len(s) for s in self.smiles]
        print(f'[NSubsetDataset] {len(self)} samples (seed={seed})')
        print(f'  SMILES lengths: min={min(lengths)}, max={max(lengths)}, '
              f'mean={np.mean(lengths):.1f}, median={np.median(lengths):.0f}')

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return {
            'ms_emb': torch.from_numpy(self.embeddings[idx]).float(),
            'smiles': self.smiles[idx],
            'maccs': smi_to_maccs(self.smiles[idx]),
        }


class SampledSubsetDataset(Dataset):
    """Sample N unique molecules (1 spectrum each) for rapid iteration.

    Preserves maximum chemical diversity by ensuring each molecule appears
    only once per epoch. ~5 min/epoch at N=40000 with batch_size=64.
    Supports both pooled ('embedding', (1024,)) and full sequence
    ('full_embedding', (60, 1024)) via embedding_key parameter.
    """

    def __init__(
        self,
        hdf5_path: str = '/root/datasets/pairs_with_embs.hdf5',
        split: str = 'train',
        n: int = 40000,
        seed: int = 42,
        embedding_key: str = 'embedding',
    ):
        # Auto-resolve local cache for full_embedding on FUSE
        if embedding_key == 'full_embedding':
            local_cache = '/tmp/full_embedding_local.h5'
            if os.path.exists(local_cache):
                hdf5_path = local_cache
        # Map split name → HDF5 value
        split_map = {'train': 0, 'val': 1, 'test': 2}
        split_val = split_map[split]

        is_full = embedding_key == 'full_embedding'

        with h5py.File(hdf5_path, 'r') as f:
            split_data = f['split'][:]
            mask = split_data == split_val

            all_smiles = [s.decode() if isinstance(s, bytes) else s
                          for s in f['smiles'][:]]

        # Filter to this split — keep track of original indices for lazy loading
        orig_indices_arr = np.where(mask)[0]
        smiles = [s for s, m in zip(all_smiles, mask) if m]
        n_split = len(smiles)
        del all_smiles, mask

        if not is_full:
            # Eager: load pooled embeddings into memory (N_split, 1024) — small
            with h5py.File(hdf5_path, 'r') as f:
                all_embs = f['embedding'][:]
            embs = all_embs[orig_indices_arr]

        # Group by unique molecule
        mol_to_indices: dict[str, list[int]] = {}
        for i, smi in enumerate(smiles):
            mol_to_indices.setdefault(smi, []).append(i)

        unique_smiles = list(mol_to_indices.keys())
        n_unique = len(unique_smiles)
        n_sample = min(n, n_unique)

        rng = np.random.RandomState(seed)
        chosen_smiles = rng.choice(unique_smiles, n_sample, replace=False)

        # Pick 1 random spectrum per chosen molecule
        self.smiles = []
        self._sample_indices = []  # indices into the split array (0..n_split-1)
        for smi in chosen_smiles:
            idx = rng.choice(mol_to_indices[smi])
            self.smiles.append(smi)
            self._sample_indices.append(idx)

        # Store embeddings or lazy handle
        if is_full:
            # Eager: load subset of full_embedding (avoids FUSE random-read latency)
            # individual ds[i] reads are fast; use sorted order for sequential disk access
            idxs = np.sort(orig_indices_arr[self._sample_indices])
            n_sample = len(idxs)
            with h5py.File(hdf5_path, 'r') as f:
                ds = f[embedding_key]
                sorted_embs = np.empty((n_sample, *ds.shape[1:]), dtype=np.float32)
                for i, row_idx in enumerate(idxs):
                    sorted_embs[i] = ds[row_idx]
            # Reorder back to match self._sample_indices order
            sort_order = np.argsort(orig_indices_arr[self._sample_indices])
            unsort_order = np.argsort(sort_order)
            self.embeddings = sorted_embs[unsort_order]
            self._h5_file = None
        else:
            # Eager: stack pooled embeddings (N_sample, 1024)
            self.embeddings = np.stack([embs[i] for i in self._sample_indices])
            self._h5_file = None

        lengths = [len(s) for s in self.smiles]
        print(f'[SampledSubsetDataset] {len(self)} unique molecules '
              f'(from {n_unique} available in {split}, seed={seed})')
        print(f'  Embedding mode: {"full_embedding (eager)" if is_full else "embedding (pooled)"}')
        print(f'  SMILES lengths: min={min(lengths)}, max={max(lengths)}, '
              f'mean={np.mean(lengths):.1f}, median={np.median(lengths):.0f}')

    def __del__(self):
        if hasattr(self, '_h5_file') and self._h5_file is not None:
            self._h5_file.close()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        if self.embeddings is not None:
            ms_emb = torch.from_numpy(self.embeddings[idx]).float()
        else:
            ms_emb = torch.from_numpy(self._emb_ds[self._orig_row_indices[idx]]).float()
        return {
            'ms_emb': ms_emb,
            'smiles': self.smiles[idx],
            'maccs': smi_to_maccs(self.smiles[idx]),
        }


class T5SmilesCollator:
    """Collate function that tokenizes SMILES with T5 tokenizer in batch."""

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        ms_embs = torch.stack([item['ms_emb'] for item in batch])
        smiles_strings = [item['smiles'] for item in batch]

        # Batch tokenize: T5 handles text → input_ids + labels internally
        # text_target creates decoder_input_ids and labels with proper shift
        encoding = self.tokenizer(
            text_target=smiles_strings,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        # T5 tokenizer returns input_ids for the target (not 'labels' key)
        labels = encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        result = {
            'ms_emb': ms_embs,
            'labels': labels,
        }

        # Pass through MACCS if present
        if 'maccs' in batch[0]:
            result['maccs'] = torch.stack([item['maccs'] for item in batch])

        return result
