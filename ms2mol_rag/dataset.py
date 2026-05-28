"""Dataset for RAG-based MS -> SMILES training.

Neighbors are pre-computed once using FAISS (fast, approximate) and cached,
so training is just a table lookup -- no KNN search per batch.

FAISS IVF100 handles 1024-d embeddings ~960x faster than sklearn's BallTree
(3.5 min vs 56 hours for 313K samples).
"""

import os
import pickle
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

CACHE_DIR = '/root/DreaMS/ms2mol_rag/outputs'


def _build_faiss_index(embeddings: np.ndarray, nlist: int = 100):
    """Build a FAISS IVF index for fast approximate nearest neighbor search.

    Args:
        embeddings: (N, D) float32 array.
        nlist: Number of IVF centroids.

    Returns:
        (index, normalized_embeddings)
    """
    import faiss
    n, d = embeddings.shape
    normalized = embeddings.astype(np.float32)
    faiss.normalize_L2(normalized)

    index = faiss.index_factory(d, f'IVF{nlist},Flat')
    if not index.is_trained:
        index.train(normalized)
    index.add(normalized)
    index.nprobe = 10  # search-time probes (trade speed for recall)
    print(f'[FAISS] IVF{nlist} index: {index.ntotal} vectors, d={d}, nprobe={index.nprobe}')
    return index, normalized


class MSSpectrumSmilesRAGDataset(Dataset):
    """Dataset with pre-computed RAG neighbor cache (FAISS-backed)."""

    def __init__(
        self,
        hdf5_path: str = '/root/datasets/pairs_with_embs.hdf5',
        split: str = 'train',
        k_contexts: int = 3,
        cache_dir: str = None,
    ):
        self.k_contexts = k_contexts
        cache_dir = cache_dir or CACHE_DIR
        self.cache_path = os.path.join(cache_dir, f'neighbors_{split}.pkl')
        self.split = split

        print(f'[Dataset] Loading {split} from {hdf5_path}')
        with h5py.File(hdf5_path, 'r') as f:
            split_data = f['split'][:]
            split_map = {'train': 0, 'val': 1, 'test': 2}
            mask = split_data == split_map[split]

            all_embs = f['embedding'][:]
            self.embeddings = all_embs[mask]
            print(f'  Embeddings: {self.embeddings.shape}')

            all_smiles = f['smiles'][:]
            self.smiles = [s.decode() if isinstance(s, bytes) else s
                           for s, m in zip(all_smiles, mask) if m]

        # Load or pre-compute neighbor cache
        self.neighbor_smiles = self._load_or_compute_neighbors()

        print(f'  Spectra: {len(self)}')

    def _load_or_compute_neighbors(self) -> list[list[str]]:
        """Load cached neighbors or compute them with FAISS."""
        if os.path.exists(self.cache_path):
            print(f'  Loading cached neighbors from {self.cache_path}')
            with open(self.cache_path, 'rb') as f:
                return pickle.load(f)

        print(f'  Pre-computing neighbors (k={self.k_contexts})...')

        if self.split == 'train':
            # Search within training set (exclude self-match)
            index, _ = _build_faiss_index(self.embeddings, nlist=100)
            n_neighbors = self.k_contexts + 1  # extra to skip self
            _, indices = index.search(self.embeddings.astype(np.float32), n_neighbors)

            neighbor_smiles = []
            for i in range(len(self.embeddings)):
                neighbors = []
                for j in range(indices.shape[1]):
                    idx = indices[i, j]
                    if idx != i:
                        neighbors.append(self.smiles[idx])
                    if len(neighbors) == self.k_contexts:
                        break
                neighbor_smiles.append(neighbors)
        else:
            # Val/test: search against training set
            print('  Loading training set for val/test retrieval...')
            with h5py.File('/root/datasets/pairs_with_embs.hdf5', 'r') as f:
                train_mask = f['split'][:] == 0
                train_embs = f['embedding'][:][train_mask]
                train_smiles = [s.decode() if isinstance(s, bytes) else s
                                for s in f['smiles'][:][train_mask]]

            index, _ = _build_faiss_index(train_embs, nlist=100)
            _, indices = index.search(self.embeddings.astype(np.float32), self.k_contexts)

            neighbor_smiles = []
            for i in range(len(self.embeddings)):
                neighbors = [train_smiles[idx] for idx in indices[i]]
                neighbor_smiles.append(neighbors)

        # Cache to pickle for future runs
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(neighbor_smiles, f)
        print(f'  Saved neighbor cache to {self.cache_path}')

        return neighbor_smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return {
            'ms_emb': torch.from_numpy(self.embeddings[idx]).float(),
            'smiles': self.smiles[idx],
            'neighbors': self.neighbor_smiles[idx],
        }


class RAGSmilesCollator:
    """Collate function with pre-computed RAG neighbors (no KNN at runtime)."""

    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        ms_embs = torch.stack([item['ms_emb'] for item in batch])
        smiles_strings = [item['smiles'] for item in batch]
        neighbor_batch = [item['neighbors'] for item in batch]

        # Tokenize target SMILES
        encoding = self.tokenizer(
            text_target=smiles_strings,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        labels = encoding['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'ms_emb': ms_embs,
            'labels': labels,
            'context_smiles': neighbor_batch,
        }
