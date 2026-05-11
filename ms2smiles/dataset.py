"""Dataset for (MS embedding, SMILES) pairs.

Supports two modes:
1. Precomputed embeddings: loads from pairs_with_embs.hdf5 (faster)
2. On-the-fly: loads raw spectra + SMILES from pairs_ready.hdf5
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from ms2smiles.config import MS2SMILESConfig


class MSSpectrumSmilesDataset(Dataset):
    """Dataset of (MS embedding, SMILES) pairs for ChemGPT fine-tuning."""

    def __init__(
        self,
        config: MS2SMILESConfig,
        split: str = 'train',  # 'train', 'val', 'test'
    ):
        self.config = config
        hdf5_path = config.data_hdf5 if config.use_precomputed_embs else config.data_raw_hdf5
        self.use_embs = config.use_precomputed_embs

        # Load tokenizer (shared with model)
        from transformers import PreTrainedTokenizerFast
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(config.chemgpt_tokenizer)

        print(f'[Dataset] Loading {split} split from {hdf5_path}')
        with h5py.File(hdf5_path, 'r') as f:
            split_data = f['split'][:]
            mask = split_data == {'train': 0, 'val': 1, 'test': 2}[split]

            if self.use_embs:
                all_embs = f['embedding'][:]
                self.embeddings = all_embs[mask]
                print(f'  Embeddings: {self.embeddings.shape}')

            all_smiles = f['smiles'][:]
            self.smiles = [s.decode() if isinstance(s, bytes) else s
                           for s, m in zip(all_smiles, mask) if m]

            # Optional: load metadata
            if 'adduct' in f:
                all_adduct = f['adduct'][:]
                self.adduct = [a.decode() if isinstance(a, bytes) else a
                               for a, m in zip(all_adduct, mask) if m]
            if 'collision_energy' in f:
                all_ce = f['collision_energy'][:]
                self.collision_energy = all_ce[mask]
            if 'precursor_mz' in f:
                all_mz = f['precursor_mz'][:]
                self.precursor_mz = all_mz[mask]

        print(f'  Spectra: {len(self)}')

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]

        # Tokenize SMILES (NO special tokens — prefix handles start signal)
        ids = self.tokenizer(smi, add_special_tokens=False)['input_ids']

        # Truncate if too long
        if len(ids) > self.config.max_seq_len:
            ids = ids[:self.config.max_seq_len]

        item = {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'labels': torch.tensor(ids, dtype=torch.long),
        }

        if self.use_embs:
            item['embedding'] = torch.from_numpy(self.embeddings[idx]).float()
        else:
            # Will need to load raw spectrum data
            raise NotImplementedError('On-the-fly mode not yet implemented')

        return item


def collate_fn(batch):
    """Collate function: pad sequences to same length in batch.

    For prefix tuning, sequences become [V_ms, token1, token2, ...].
    Padding is applied only to the SMILES token portion.
    """
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]

    # Pad sequences to same length (pad_token_id=1 for [PAD])
    padded_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    # Attention mask: 1 for real tokens, 0 for padding
    attention_mask = (padded_ids != 1).long()

    result = {
        'smiles_ids': padded_ids,
        'labels': padded_labels,
        'attention_mask': attention_mask,
    }

    # Embeddings: stack if present
    if 'embedding' in batch[0]:
        result['embeddings'] = torch.stack([item['embedding'] for item in batch])

    return result
