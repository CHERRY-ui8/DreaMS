"""Dataset for MS ↔ molecule contrastive retrieval.

Data flow:
    1. Load (1024-d MS embedding, SMILES, split) from pairs_with_embs.hdf5
    2. Pre-compute MoLFormer embeddings for ALL unique SMILES in one pass
    3. At __getitem__ time: return (ms_emb, mol_emb, label_idx)

This avoids re-running MoLFormer every epoch (44M params → ~3.5 it/s on CPU).
Pre-computing 46K unique molecules takes ~10 minutes, done once on init.
"""

import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

# HF mirror (port 443 is blocked, mirror works)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/t5cache'

MOLFORMER_NAME = 'ibm/MoLFormer-XL-both-10pct'
CACHE_DIR = '/root/DreaMS/ms2mol_retrieval/outputs'


def _load_molformer(device: str = 'cuda') -> tuple[AutoModel, AutoTokenizer]:
    """Load frozen MoLFormer model on target device.

    Sets deterministic=True at config level BEFORE model construction to
    avoid GPU-side QR decomposition in Performer attention (which triggers
    CUSOLVER_STATUS_INTERNAL_ERROR on this CUDA/PyTorch version).

    Returns:
        (model, tokenizer) — both on target device, model in eval mode.
    """
    print(f'[MoLFormer] Loading {MOLFORMER_NAME} on {device}...')
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(
        MOLFORMER_NAME, trust_remote_code=True,
    )
    config.deterministic = True       # prevents QR regeneration during training
    config.deterministic_eval = True  # prevents QR regeneration during eval
    tokenizer = AutoTokenizer.from_pretrained(
        MOLFORMER_NAME, trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        MOLFORMER_NAME, trust_remote_code=True, config=config,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[MoLFormer] Loaded: {n_params:,} params (frozen)')
    return model, tokenizer


@torch.no_grad()
def _compute_molformer_embeddings(
    smiles_list: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    batch_size: int = 256,
    device: str = 'cuda',
) -> np.ndarray:
    """Compute MoLFormer [CLS] embeddings for all SMILES.

    Uses the [CLS] token embedding (first position) as the molecule representation,
    same as the MoLFormer paper.

    Args:
        smiles_list: List of SMILES strings.
        model: MoLFormer model (eval mode, frozen).
        tokenizer: Corresponding tokenizer.
        batch_size: Batch size for encoding.
        device: Target device.
    Returns:
        (N, 768) float32 numpy array of [CLS] embeddings.
    """
    all_embs = []
    for start in range(0, len(smiles_list), batch_size):
        batch = smiles_list[start:start + batch_size]
        inputs = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors='pt',
        ).to(device)
        outputs = model(**inputs)
        # MoLFormer returns (B, L, 768); [CLS] is at position 0
        cls_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # (B, 768)
        all_embs.append(cls_embs)

    return np.concatenate(all_embs, axis=0).astype(np.float32)


class MSMolRetrievalDataset(Dataset):
    """Dataset for contrastive retrieval with pre-computed MoLFormer embeddings.

    Pre-computes MoLFormer embeddings for all unique SMILES in the dataset once
    during initialization, caching them to disk for future runs.

    Args:
        hdf5_path: Path to pairs_with_embs.hdf5.
        split: 'train', 'val', or 'test'.
        cache_dir: Directory for caching MoLFormer embeddings.
        force_recompute: If True, recompute MoLFormer embeddings even if cached.
    """

    def __init__(
        self,
        hdf5_path: str = '/root/datasets/pairs_with_embs.hdf5',
        split: str = 'train',
        cache_dir: str = None,
        force_recompute: bool = False,
    ):
        cache_dir = cache_dir or CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)

        split_map = {'train': 0, 'val': 1, 'test': 2}
        split_id = split_map[split]
        self.split = split

        # ── Load data from HDF5 ──
        print(f'[Dataset] Loading {split} split from {hdf5_path}')
        with h5py.File(hdf5_path, 'r') as f:
            split_col = f['split'][:]
            mask = split_col == split_id

            self.ms_embs = f['embedding'][:][mask].astype(np.float32)  # (N, 1024)
            self.smiles_list = [
                s.decode() if isinstance(s, bytes) else s
                for s in f['smiles'][:][mask]
            ]
            print(f'[Dataset]  MS embeddings: {self.ms_embs.shape}')
            print(f'[Dataset]  SMILES: {len(self.smiles_list)} samples')

        # ── Build molecule ID mapping ──
        # Each unique SMILES gets an integer ID. For contrastive learning,
        # the label is the molecule index (multiple spectra per molecule
        # share the same molecule ID).
        unique_smiles = sorted(set(self.smiles_list))
        self.smiles_to_id = {s: i for i, s in enumerate(unique_smiles)}
        self.mol_ids = np.array(
            [self.smiles_to_id[s] for s in self.smiles_list],
            dtype=np.int64,
        )
        print(f'[Dataset]  Unique molecules: {len(unique_smiles)}')

        # ── Compute molecular clusters via Morgan Fingerprint + FAISS K-means ──
        # This is used by HardNegativeBatchSampler to create hard in-batch negatives.
        self.cluster_ids = self._compute_or_load_clusters(
            unique_smiles, self.smiles_to_id, cache_dir, force_recompute,
        )

        # ── Pre-compute / load MoLFormer embeddings ──
        cache_path = os.path.join(cache_dir, f'molformer_embs_{split}.npy')
        id_map_path = os.path.join(cache_dir, f'molformer_ids_{split}.pkl')

        if force_recompute or not os.path.exists(cache_path):
            print(f'[Dataset]  Computing MoLFormer embeddings for {len(unique_smiles)} unique molecules...')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model, tokenizer = _load_molformer(device)
            unique_features = _compute_molformer_embeddings(
                unique_smiles, model, tokenizer, device=device,
            )
            # Cache to disk
            np.save(cache_path, unique_features)
            with open(id_map_path, 'wb') as f:
                pickle.dump(self.smiles_to_id, f)
            print(f'[Dataset]  Saved MoLFormer embeddings to {cache_path}')
            # Clean up model to free GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f'[Dataset]  Loading cached MoLFormer embeddings from {cache_path}')
            unique_features = np.load(cache_path)

        # Build lookup: for each sample index, the MoLFormer embedding of its molecule
        self.mol_embs = np.zeros((len(self), 768), dtype=np.float32)
        for i, s in enumerate(self.smiles_list):
            mol_id = self.smiles_to_id[s]
            self.mol_embs[i] = unique_features[mol_id]

        print(f'[Dataset]  MoLFormer embeddings: {self.mol_embs.shape}')
        print(f'[Dataset]  Total: {len(self)} samples')

    def _compute_or_load_clusters(
        self, unique_smiles: list, smiles_to_id: dict,
        cache_dir: str, force_recompute: bool,
    ) -> np.ndarray:
        """Compute Morgan fingerprint clusters (FAISS K-means, n_clusters=1000).

        Clusters molecules by structural similarity (Tanimoto of Morgan fingerprints).
        Returns cluster_id for EACH SAMPLE in the dataset (not just unique molecules).

        Cached to disk as npy/pkl for reuse.
        """
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        from sklearn.cluster import MiniBatchKMeans
        RDLogger.DisableLog('rdApp.*')  # suppress MorganGenerator deprecation

        cache_cluster = os.path.join(cache_dir, 'cluster_ids.npy')
        cache_unique = os.path.join(cache_dir, 'unique_clusters.npy')

        if not force_recompute and os.path.exists(cache_cluster):
            print(f'[Clusters] Loading cached cluster IDs from {cache_cluster}')
            return np.load(cache_cluster)

        # Compute Morgan fingerprints for all unique molecules
        print(f'[Clusters] Computing Morgan fingerprints for {len(unique_smiles)} unique molecules...')
        fps_list = [self._morgan_fp(smi, 2048) for smi in unique_smiles]
        fp_matrix = np.stack(fps_list)  # (N_unique, 2048)

        # MiniBatchKMeans clustering (much faster than FAISS CPU K-means for high-d data)
        n_clusters = min(1000, len(fp_matrix) // 10)
        print(f'[Clusters] MiniBatchKMeans: {len(fp_matrix)} molecules → {n_clusters} clusters')
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=42, batch_size=16384,
            n_init=1, max_iter=50, verbose=1, init='random',
        )
        unique_cluster_ids = kmeans.fit_predict(fp_matrix).astype(np.int64)

        # Map each SAMPLE to its molecule's cluster_id
        mol_to_cluster = {}
        for uniq_idx, smi in enumerate(unique_smiles):
            mid = smiles_to_id[smi]
            mol_to_cluster[mid] = unique_cluster_ids[uniq_idx]

        cluster_ids = np.array([mol_to_cluster[mid] for mid in self.mol_ids.tolist()],
                               dtype=np.int64)

        # Cache
        np.save(cache_cluster, cluster_ids)
        np.save(cache_unique, unique_cluster_ids)
        unique_count = len(set(unique_cluster_ids.tolist()))
        print(f'[Clusters] Saved to {cache_cluster}')
        print(f'[Clusters]  {unique_count}/{n_clusters} clusters non-empty')

        return cluster_ids

    @staticmethod
    def _morgan_fp(smiles: str, nbits: int = 2048) -> np.ndarray:
        """Compute Morgan fingerprint as numpy array (binary)."""
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nbits)
            return np.array(fp, dtype=np.float32)
        return np.zeros(nbits, dtype=np.float32)

    def __len__(self):
        return len(self.ms_embs)

    def __getitem__(self, idx: int) -> dict:
        return {
            'ms_emb': torch.from_numpy(self.ms_embs[idx]).float(),
            'mol_emb': torch.from_numpy(self.mol_embs[idx]).float(),
            'mol_id': torch.tensor(self.mol_ids[idx], dtype=torch.long),
            'smiles': self.smiles_list[idx],
        }


# ═══════════════════════════════════════════════════════════════════
# MultiTaskRetrievalDataset
# ═══════════════════════════════════════════════════════════════════

class MultiTaskRetrievalDataset(Dataset):
    """多任务数据集：加载 ms_emb + MoLFormer 特征 + MACCS + 分子量

    数据来源：
        - ms_emb:      HDF5 的 'embedding' 字段 (1024-d)
        - mol_emb:     预计算 MoLFormer 缓存 (768-d)
        - maccs:       HDF5 的 'maccs' 字段 (167-d，取前 166 列)
        - mol_weight:  从 SMILES 用 RDKit 计算，预缓存

    Args:
        hdf5_path:      HDF5 数据路径
        split:          'train', 'val', 或 'test'
        cache_dir:      输出目录（存放分子量缓存）
        molformer_cache_dir: MoLFormer 缓存目录（默认 shared_cache）
    """

    def __init__(
        self,
        hdf5_path: str = '/root/datasets/pairs_with_embs.hdf5',
        split: str = 'train',
        cache_dir: str = None,
        molformer_cache_dir: str = None,
        mw_mean: float = None,
        mw_std: float = None,
    ):
        cache_dir = cache_dir or '/root/DreaMS/ms2mol_retrieval/shared_cache'
        molformer_cache_dir = molformer_cache_dir or '/root/DreaMS/ms2mol_retrieval/shared_cache'
        split_map = {'train': 0, 'val': 1, 'test': 2}
        split_id = split_map[split]
        self.split = split

        # ── 从 HDF5 加载 ──
        print(f'[MultiTaskDataset] Loading {split} split from {hdf5_path}')
        with h5py.File(hdf5_path, 'r') as f:
            split_col = f['split'][:]
            mask = split_col == split_id
            self.ms_embs = f['embedding'][:][mask].astype(np.float32)       # (N, 1024)
            self.maccs = f['maccs'][:][mask][:, :166].astype(np.float32)    # (N, 166) — 去掉第 167 列 padding
            self.smiles_list = [
                s.decode() if isinstance(s, bytes) else s
                for s in f['smiles'][:][mask]
            ]
            print(f'[MultiTaskDataset]  ms_emb: {self.ms_embs.shape}')
            print(f'[MultiTaskDataset]  maccs:  {self.maccs.shape}')
            print(f'[MultiTaskDataset]  smiles: {len(self.smiles_list)} samples')

        # ── 加载 MoLFormer 缓存 ──
        cache_path = os.path.join(molformer_cache_dir, f'molformer_embs_{split}.npy')
        id_map_path = os.path.join(molformer_cache_dir, f'molformer_ids_{split}.pkl')
        if os.path.exists(cache_path) and os.path.exists(id_map_path):
            print(f'[MultiTaskDataset]  Loading MoLFormer cache: {cache_path}')
            import pickle
            unique_features = np.load(cache_path)
            with open(id_map_path, 'rb') as f:
                smiles_to_molid = pickle.load(f)
            self.mol_embs = np.zeros((len(self), 768), dtype=np.float32)
            for i, s in enumerate(self.smiles_list):
                mid = smiles_to_molid[s]
                self.mol_embs[i] = unique_features[mid]
            print(f'[MultiTaskDataset]  mol_embs: {self.mol_embs.shape}')
        else:
            print(f'[MultiTaskDataset]  ⚠ MoLFormer 缓存未找到: {cache_path}')
            print(f'[MultiTaskDataset]  回退到随机特征（仅用于调试）')
            self.mol_embs = np.random.randn(len(self), 768).astype(np.float32)

        # ── 分子量：预计算并缓存 ──
        mw_cache = os.path.join(cache_dir, f'mol_weight_{split}.npy')
        if os.path.exists(mw_cache):
            print(f'[MultiTaskDataset]  Loading mol_weight cache: {mw_cache}')
            self.mol_weights = np.load(mw_cache)  # (N, 1)
        else:
            print(f'[MultiTaskDataset]  Computing mol_weight from SMILES...')
            weights = []
            from rdkit import Chem
            from rdkit.Chem import Descriptors
            for s in self.smiles_list:
                mol = Chem.MolFromSmiles(s)
                w = Descriptors.ExactMolWt(mol) if mol is not None else 0.0
                weights.append([w])
            self.mol_weights = np.array(weights, dtype=np.float32)
            os.makedirs(cache_dir, exist_ok=True)
            np.save(mw_cache, self.mol_weights)
            print(f'[MultiTaskDataset]  Saved mol_weight to {mw_cache}')
        print(f'[MultiTaskDataset]  mol_weight: {self.mol_weights.shape}, '
              f'range [{self.mol_weights.min():.1f}, {self.mol_weights.max():.1f}] Da')

        # ── 分子 ID 映射（用于 FAISS 检索评估） ──
        unique_smiles = sorted(set(self.smiles_list))
        self.smiles_to_mol_id = {s: i for i, s in enumerate(unique_smiles)}
        self.mol_ids = np.array(
            [self.smiles_to_mol_id[s] for s in self.smiles_list],
            dtype=np.int64,
        )
        print(f'[MultiTaskDataset]  Unique molecules: {len(unique_smiles)}')

        # ── 分子量归一化参数（val/test 复用训练集统计量，避免 data leakage） ──
        if mw_mean is not None and mw_std is not None:
            self.mw_mean = mw_mean
            self.mw_std = mw_std
            print(f'[MultiTaskDataset]  mol_weight norm: using train stats mean={self.mw_mean:.2f}, std={self.mw_std:.2f}')
        else:
            self.mw_mean = float(self.mol_weights.mean())
            self.mw_std = float(max(self.mol_weights.std(), 1.0))
            print(f'[MultiTaskDataset]  mol_weight norm: computed mean={self.mw_mean:.2f}, std={self.mw_std:.2f}')

        print(f'[MultiTaskDataset]  Total: {len(self)} samples')

    def __len__(self):
        return len(self.ms_embs)

    def __getitem__(self, idx: int) -> dict:
        # 分子量归一化为 z-score
        mw = (self.mol_weights[idx][0] - self.mw_mean) / self.mw_std
        return {
            'ms_emb': torch.from_numpy(self.ms_embs[idx]).float(),          # (1024,)
            'mol_emb': torch.from_numpy(self.mol_embs[idx]).float(),         # (768,)
            'maccs': torch.from_numpy(self.maccs[idx]).float(),              # (166,)
            'mol_weight': torch.tensor(mw, dtype=torch.float).unsqueeze(0),  # (1,) z-score
            'mol_id': torch.tensor(self.mol_ids[idx], dtype=torch.long),     # () scalar
        }
