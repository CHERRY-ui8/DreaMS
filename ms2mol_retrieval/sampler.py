"""Hard-negative batch sampler for contrastive learning.

Strategy: Force structurally similar molecules (same Morgan fingerprint cluster)
into the same batch. This makes in-batch negatives MUCH harder — the model
must distinguish between molecularly similar compounds, not just random ones.

Pipeline:
    1. Offline: RDKit Morgan fingerprints → FAISS K-means → cluster_id per molecule
    2. At each batch: sample hard_size from one cluster + random_size globally
"""

import math
import numpy as np
from torch.utils.data import Sampler, DataLoader


class HardNegativeBatchSampler(Sampler):
    """Batch sampler with controlled hard negative ratio.

    Each batch contains:
        - hard_size: samples from the SAME molecular cluster (structurally similar)
        - random_size: globally random samples (maintain diversity)

    Args:
        cluster_ids: (N,) numpy array of cluster IDs for each sample in the dataset.
        batch_size: Total batch size.
        hard_ratio: Fraction of batch from the same cluster (default: 0.25).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        cluster_ids: np.ndarray,
        batch_size: int,
        hard_ratio: float = 0.25,
        seed: int = 42,
    ):
        assert 0.0 < hard_ratio <= 1.0, 'hard_ratio must be in (0, 1]'
        self.cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
        self.batch_size = batch_size
        self.hard_size = max(1, int(batch_size * hard_ratio))
        self.random_size = batch_size - self.hard_size
        self.num_samples = len(self.cluster_ids)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Build cluster → indices mapping
        unique_clusters = sorted(set(cluster_ids))
        self.clusters = list(unique_clusters)
        self.cluster_to_indices = {}
        for c in unique_clusters:
            self.cluster_to_indices[c] = np.where(self.cluster_ids == c)[0]
        # Pre-filter clusters that have < 2 samples (can't form hard negatives)
        self.usable_clusters = [c for c in self.clusters
                                if len(self.cluster_to_indices[c]) >= 2]

        n_batches = self.num_samples // self.batch_size
        n_hard_per_epoch = n_batches * self.hard_size
        n_random_per_epoch = n_batches * self.random_size
        print(f'[HardNegativeSampler] batch={batch_size}, '
              f'hard_ratio={hard_ratio:.2f} ({self.hard_size}+{self.random_size})')
        print(f'[HardNegativeSampler] clusters={len(self.clusters)}, '
              f'usable={len(self.usable_clusters)}, '
              f'batches/epoch={n_batches}')

    def __iter__(self):
        n_batches = self.num_samples // self.batch_size

        for _ in range(n_batches):
            # 1. Hard negatives: pick ONE random cluster, sample from it
            if len(self.usable_clusters) > 0:
                cluster = self.rng.choice(self.usable_clusters)
                indices = self.cluster_to_indices[cluster]
                replace = len(indices) < self.hard_size
                hard_batch = self.rng.choice(
                    indices, self.hard_size, replace=replace,
                ).tolist()
            else:
                # Fallback: all random
                hard_batch = self.rng.choice(
                    self.num_samples, self.hard_size, replace=False,
                ).tolist()

            # 2. Random negatives: globally random
            random_batch = self.rng.choice(
                self.num_samples, self.random_size, replace=False,
            ).tolist()

            # 3. Combine and shuffle
            batch = hard_batch + random_batch
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_samples // self.batch_size
