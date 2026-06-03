"""CLIP-style dual encoder for MS spectrum ↔ molecule retrieval.

Architecture:
    MS:   DreaMS (frozen, 1024-d) → Projector(1024→256) → L2Norm → 256-d
    Mol:  MoLFormer (frozen, 768-d) → Projector(768→256) → L2Norm → 256-d
    Loss: InfoNCE with learnable logit_scale (temperature)

Reference: CLIP (Radford et al., 2021) — dual encoder + contrastive loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """Configurable-depth MLP projection head with LayerNorm.

    Architecture:
        Depth=2: Linear(in, hidden) → GELU → LayerNorm → Linear(hidden, out)
        Depth=3: Linear(in, hidden) → GELU → LayerNorm → Linear(hidden, hidden) → GELU → LayerNorm → Linear(hidden, out)

    The LayerNorm before final projection stabilizes training and avoids
    representation collapse (SimCLR v2 insight).

    Args:
        in_dim: Input dimension.
        hidden_dim: Hidden layer width (default: 1024 for more capacity).
        out_dim: Output projection dimension.
        depth: Number of hidden layers (2 or 3, default: 2 for backward compat).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 1024,
                 out_dim: int = 256, depth: int = 2):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        ]
        # Add extra hidden layers
        for _ in range(depth - 2):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            ])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MSMolCLIP(nn.Module):
    """CLIP-style dual encoder for MS ↔ molecule contrastive learning.

    Both base encoders (DreaMS, MoLFormer) are frozen — only projection heads
    and logit_scale are trained. The model learns a joint 256-d embedding space
    where paired (MS, molecule) vectors are close and unpaired ones are far.

    Args:
        ms_dim: DreaMS embedding dimension (default: 1024).
        mol_dim: MoLFormer embedding dimension (default: 768).
        proj_dim: Shared projection dimension (default: 256).
        proj_hidden: Projector hidden layer width (default: 1024).
        proj_depth: Number of hidden layers in projector (2 or 3, default: 2).
    """

    def __init__(
        self,
        ms_dim: int = 1024,
        mol_dim: int = 768,
        proj_dim: int = 256,
        proj_hidden: int = 1024,
        proj_depth: int = 2,
    ):
        super().__init__()
        self.ms_dim = ms_dim
        self.mol_dim = mol_dim
        self.proj_dim = proj_dim
        self.proj_depth = proj_depth
        self.proj_hidden = proj_hidden

        # Projection heads
        self.ms_projector = Projector(ms_dim, proj_hidden, proj_dim, proj_depth)
        self.mol_projector = Projector(mol_dim, proj_hidden, proj_dim, proj_depth)

        # Learnable temperature (logit_scale)
        # Initialized to 1/0.07 ≈ 14.29 as in CLIP
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / 0.07)))

        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[MSMolCLIP] Total params: {n_total:,} (trainable: {n_train:,})')
        print(f'[MSMolCLIP] MS: {ms_dim} → {proj_dim} | Mol: {mol_dim} → {proj_dim}')

    def encode_ms(self, ms_emb: torch.Tensor) -> torch.Tensor:
        """Encode MS embeddings to normalized joint space.

        Args:
            ms_emb: (B, ms_dim) — DreaMS position-0 embeddings.
        Returns:
            (B, proj_dim) — L2-normalized MS features.
        """
        emb = self.ms_projector(ms_emb)
        return F.normalize(emb, dim=-1)

    def encode_mol(self, mol_emb: torch.Tensor) -> torch.Tensor:
        """Encode molecule embeddings to normalized joint space.

        Args:
            mol_emb: (B, mol_dim) — MoLFormer [CLS] embeddings.
        Returns:
            (B, proj_dim) — L2-normalized molecule features.
        """
        emb = self.mol_projector(mol_emb)
        return F.normalize(emb, dim=-1)

    def forward(
        self,
        ms_emb: torch.Tensor,
        mol_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through both encoders.

        Args:
            ms_emb: (B, ms_dim) — MS embeddings.
            mol_emb: (B, mol_dim) — Molecule embeddings.
        Returns:
            (ms_feat, mol_feat): Both (B, proj_dim), L2-normalized.
        """
        return self.encode_ms(ms_emb), self.encode_mol(mol_emb)

    def compute_loss(
        self,
        ms_feat: torch.Tensor,
        mol_feat: torch.Tensor,
    ) -> dict:
        """Compute symmetric InfoNCE loss (CLIP-style).

        Logits = (ms_feat @ mol_feat.T) * logit_scale.exp()

        For each MS feature, the correct molecule is at position i (diagonal).
        Loss is computed symmetrically:
            loss_ms = CE(MS→Mol similarities, correct indices)
            loss_mol = CE(Mol→MS similarities, correct indices)
            total = (loss_ms + loss_mol) / 2

        Args:
            ms_feat: (B, proj_dim), L2-normalized MS features.
            mol_feat: (B, proj_dim), L2-normalized molecule features.
        Returns:
            dict with 'loss', 'acc_ms', 'acc_mol', 'logit_scale'.
        """
        B = ms_feat.size(0)

        # Logit scale (learnable temperature)
        scale = self.logit_scale.exp()
        logits = ms_feat @ mol_feat.T * scale  # (B, B)

        # Labels: diagonal is the correct pair
        labels = torch.arange(B, device=logits.device, dtype=torch.long)

        # Symmetric loss
        loss_ms = F.cross_entropy(logits, labels)
        loss_mol = F.cross_entropy(logits.T, labels)
        loss = (loss_ms + loss_mol) / 2.0

        # In-batch accuracy
        with torch.no_grad():
            pred_ms = logits.argmax(dim=-1)  # (B,) — for each MS, which mol?
            pred_mol = logits.argmax(dim=0)  # (B,) — for each mol, which MS?
            acc_ms = (pred_ms == labels).float().mean().item()
            acc_mol = (pred_mol == labels).float().mean().item()

            # Top-5 recall
            top5 = logits.topk(5, dim=-1).indices  # (B, 5)
            recall_ms_5 = (top5 == labels.unsqueeze(1)).any(dim=1).float().mean().item()
            top5_mol = logits.topk(5, dim=0).indices  # (5, B) for mol→MS
            recall_mol_5 = (top5_mol == labels.unsqueeze(0)).any(dim=0).float().mean().item()

        return {
            'loss': loss,
            'acc_ms': acc_ms,
            'acc_mol': acc_mol,
            'acc': (acc_ms + acc_mol) / 2.0,
            'recall_ms@5': recall_ms_5,
            'recall_mol@5': recall_mol_5,
            'logit_scale': scale.item(),
        }

    @torch.no_grad()
    def compute_retrieval_metrics(
        self,
        ms_feat_all: torch.Tensor,
        mol_feat_all: torch.Tensor,
        ms_labels: list[int],
        mol_labels: list[int],
        ks: list[int] = None,
        device: str = 'cuda',
    ) -> dict:
        """Full-database retrieval recall via FAISS.

        Args:
            ms_feat_all: (N_ms, proj_dim) — all MS features.
            mol_feat_all: (N_mol, proj_dim) — all unique molecule features.
            ms_labels: (N_ms,) — integer label for each MS (maps to molecule index).
            mol_labels: (N_mol,) — integer label for each molecule.
            ks: List of K values for recall@K (default: [1, 5, 10]).
            device: Target device for computation.
        Returns:
            dict with recall@{k} for each k.
        """
        if ks is None:
            ks = [1, 5, 10]
        import faiss
        import numpy as np

        ms_np = ms_feat_all.cpu().numpy().astype(np.float32)
        mol_np = mol_feat_all.cpu().numpy().astype(np.float32)
        ms_labels_np = np.array(ms_labels, dtype=np.int64)
        mol_labels_np = np.array(mol_labels, dtype=np.int64)

        # Build FAISS index over molecule embeddings
        d = mol_np.shape[1]
        index = faiss.IndexFlatIP(d)  # Inner product (cosine since L2-normalized)
        index.add(mol_np)

        # Search: for each MS, find top-k molecules
        top_k = max(ks)
        distances, indices = index.search(ms_np, top_k)  # (N_ms, top_k)

        # For each MS query, check if its correct molecule is in top-k
        recall = {}
        for k in ks:
            hits = 0
            for i in range(len(ms_np)):
                query_label = ms_labels_np[i]
                retrieved_labels = mol_labels_np[indices[i, :k]]
                if query_label in retrieved_labels:
                    hits += 1
            recall[f'recall@{k}'] = hits / len(ms_np)

        return recall


class MSMolCLIPMultiTask(MSMolCLIP):
    """CLIP-style 512-d + MACCS + 分子量 多任务模型。

    在 MSMolCLIP 的 512-d 双投影基础上，增加两个辅助任务 head：
    - MACCS 子结构预测 (1024→256→166)
    - 分子量预测 (1024→64→1)

    两个辅助 head 直接从 1024-d 输入特征出发，不影响 InfoNCE 投影空间。

    Args:
        ms_dim: DreaMS embedding 维度 (default: 1024).
        mol_dim: MoLFormer embedding 维度 (default: 768).
        proj_dim: 共享投影空间维度 (default: 512).
        proj_hidden: 投影头隐藏层宽度 (default: 1024).
        proj_depth: 投影头深度 2 或 3 (default: 2).
        dropout: 辅助 head 的 dropout (default: 0.1).
    """

    def __init__(
        self,
        ms_dim: int = 1024,
        mol_dim: int = 768,
        proj_dim: int = 512,
        proj_hidden: int = 1024,
        proj_depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(ms_dim, mol_dim, proj_dim, proj_hidden, proj_depth)

        # ── 辅助任务 heads（从 1024-d 输入出发） ──
        self.proj_maccs = nn.Sequential(
            nn.Linear(ms_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 166),
        )
        self.proj_mol_weight = nn.Sequential(
            nn.Linear(ms_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # 重新统计参数
        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[MSMolCLIPMultiTask] Total params: {n_total:,} (trainable: {n_train:,})')

    def forward(
        self, ms_emb: torch.Tensor, mol_emb: torch.Tensor
    ) -> dict:
        """前向传播。

        Returns:
            dict:
                'ms_feat': (B, proj_dim) L2-normed MS 特征
                'mol_feat': (B, proj_dim) L2-normed MoL 特征
                'maccs_logits': (B, 166) MACCS logits
                'mol_weight': (B, 1) 分子量预测
        """
        ms_feat = self.encode_ms(ms_emb)
        mol_feat = self.encode_mol(mol_emb)
        return {
            'ms_feat': ms_feat,
            'mol_feat': mol_feat,
            'maccs_logits': self.proj_maccs(ms_emb),
            'mol_weight': self.proj_mol_weight(ms_emb),
        }


class MSMolCLIPSharedTrunk(MSMolCLIP):
    """MSMolCLIPMultiTask with a shared trunk before task heads.

    Architecture:
        ms_emb (1024) ──→ SharedTrunk ──→ shared_feat (512) ──┬──→ CrossHead → 512-d L2Norm → InfoNCE
                                                               ├──→ MACCSHead → 166 → BCE
                                                               └──→ MWHead     → 1   → Huber
        mol_emb (768) ──→ MoL Projector (unchanged) ──→ 512-d L2Norm ──────────────────────┘

    The shared trunk receives gradients from all three tasks, enabling MACCS/MW
    supervision to influence the cross-modal retrieval projection.

    Args:
        ms_dim: DreaMS embedding dimension (default: 1024).
        mol_dim: MoLFormer embedding dimension (default: 768).
        proj_dim: Shared projection dimension (default: 512).
        proj_hidden: Projector hidden layer width (default: 1024).
        proj_depth: Projector depth 2 or 3 (default: 2).
        trunk_dim: Shared trunk output dimension (default: 512).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        ms_dim: int = 1024,
        mol_dim: int = 768,
        proj_dim: int = 512,
        proj_hidden: int = 1024,
        proj_depth: int = 2,
        trunk_dim: int = 512,
        dropout: float = 0.1,
    ):
        # Call MSMolCLIP's __init__ (NOT MSMolCLIPMultiTask) to set up
        # ms_projector, mol_projector, logit_scale
        super().__init__(ms_dim, mol_dim, proj_dim, proj_hidden, proj_depth)

        # ── Shared trunk: ms_emb (1024) → trunk_dim ──
        self.shared_trunk = nn.Sequential(
            nn.Linear(ms_dim, trunk_dim),
            nn.LayerNorm(trunk_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Cross-modal head: trunk_dim → proj_dim (L2-normed for InfoNCE) ──
        # This replaces ms_projector's role for retrieval
        self.cross_head = nn.Sequential(
            nn.Linear(trunk_dim, proj_hidden),
            nn.GELU(),
            nn.LayerNorm(proj_hidden),
            nn.Dropout(dropout),
            nn.Linear(proj_hidden, proj_dim),
        )

        # ── MACCS head: trunk_dim → 256 → 166 ──
        self.proj_maccs = nn.Sequential(
            nn.Linear(trunk_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 166),
        )

        # ── MW head: trunk_dim → 64 → 1 ──
        self.proj_mol_weight = nn.Sequential(
            nn.Linear(trunk_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        n_total = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[MSMolCLIPSharedTrunk] Total params: {n_total:,} (trainable: {n_train:,})')
        print(f'[MSMolCLIPSharedTrunk] Trunk: {ms_dim} → {trunk_dim} → {proj_dim} | '
              f'MACCS: {trunk_dim}→256→166 | MW: {trunk_dim}→64→1')

    def forward(
        self, ms_emb: torch.Tensor, mol_emb: torch.Tensor
    ) -> dict:
        """Forward pass.

        Returns:
            dict with 'ms_feat', 'mol_feat', 'maccs_logits', 'mol_weight'.
        """
        # Shared trunk: all three heads receive the same shared representation
        shared = self.shared_trunk(ms_emb)

        # Cross-modal (InfoNCE)
        ms_feat = F.normalize(self.cross_head(shared), dim=-1)

        # Molecule side: unchanged
        mol_feat = self.encode_mol(mol_emb)

        return {
            'ms_feat': ms_feat,
            'mol_feat': mol_feat,
            'maccs_logits': self.proj_maccs(shared),
            'mol_weight': self.proj_mol_weight(shared),
        }
