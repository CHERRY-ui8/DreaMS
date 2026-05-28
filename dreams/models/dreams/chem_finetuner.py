"""
DreaMS_ChemFinetuner — Continual Pre-training (Multi-task) of DreaMS encoder.

Wraps the pre-trained DreaMS with a MACCS multi-label classification head
while retaining the original Masked Peak Prediction objective.

Architecture:
    DreaMS encoder (frozen/learnable) → (B, 60, 1024) full sequence
        ├── masked_peak_head (reused) → masked_peak_loss (m/z + intensity prediction)
        └── MACCS head (new)           → maccs_loss (167-bit fingerprint classification)

Total loss = 0.2 × masked_peak_loss + 1.0 × maccs_loss

The full (B, 60, 1024) sequence is returned for downstream Q-Former / T5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# DreaMS utilities for hot-encoding ground truth
from dreams.utils.spectra import to_hot


class DreaMS_ChemFinetuner(nn.Module):
    """Wrapper for continual pre-training of DreaMS with MACCS supervision.

    Args:
        dreams_encoder: Pre-trained DreaMS (pl.LightningModule) instance.
            Must have attributes: d_model, train_objective, dformat, hot_mz_bin_size,
            ff_out, ff_out_intens (if mask_peak_hot), mz_masking_loss.
        pooling: How to aggregate (B, 60, 1024) → (B, feat_dim) for MACCS head.
            'cls' —  only [CLS] token (position 0).          feat_dim = 1024
            'mean' — mean over all 60 positions.              feat_dim = 1024
            'hierarchical' — [CLS] cat mean(fragment peaks). feat_dim = 2048
    """

    def __init__(self, dreams_encoder: nn.Module, pooling: str = 'cls'):
        super().__init__()
        self.encoder = dreams_encoder
        self.pooling = pooling
        d_model = self.encoder.d_model  # 1024

        assert pooling in ('cls', 'mean', 'hierarchical'), \
            f"pooling must be 'cls', 'mean', or 'hierarchical', got '{pooling}'"

        # Input dimension depends on pooling strategy
        if pooling == 'hierarchical':
            in_dim = d_model * 2  # 2048: [CLS] + mean of fragment peaks
        else:
            in_dim = d_model      # 1024: [CLS] only or mean of all

        # ── MACCS head: 2-layer MLP ──
        self.maccs_head = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 167),
        )

        self._init_weights()

    def _pool_sequence(self, sequence_output: torch.Tensor) -> torch.Tensor:
        """Aggregate (B, 60, 1024) → (B, feat_dim) according to self.pooling."""
        if self.pooling == 'cls':
            # Only the precursor token (position 0)
            return sequence_output[:, 0, :]                       # (B, 1024)

        if self.pooling == 'mean':
            # Uniform average over all 60 peak positions
            return sequence_output.mean(dim=1)                     # (B, 1024)

        if self.pooling == 'hierarchical':
            # Precursor token + average of fragment peaks (positions 1..59)
            cls = sequence_output[:, 0, :]                        # (B, 1024)
            frag = sequence_output[:, 1:, :].mean(dim=1)          # (B, 1024)
            return torch.cat([cls, frag], dim=-1)                  # (B, 2048)

    def _init_weights(self):
        """Xavier init for MACCS head; keep DreaMS weights unchanged."""
        for m in self.maccs_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        spec_mask: torch.Tensor,
        spec_real: torch.Tensor,
        mask: torch.Tensor,
        maccs_labels: torch.Tensor,
        charge: torch.Tensor = None,
    ) -> dict:
        """Forward pass combining masked peak SSL + MACCS classification.

        Args:
            spec_mask:  (B, 60, 2)  Spectrum with some peaks masked.
            spec_real:  (B, 60, 2)  Original unmasked spectrum (ground truth).
            mask:       (B, 60)     Boolean, True = positions that are masked.
            maccs_labels: (B, 167)  MACCS key binary targets.
            charge:     (B,) optional charge values.

        Returns:
            dict with:
                total_loss:        scalar, combined loss
                masked_peak_loss:  scalar, SSL m/z + intensity prediction loss
                maccs_loss:        scalar, MACCS BCE loss
                sequence_output:   (B, 60, 1024) full sequence embeddings
                maccs_logits:      (B, 167) raw logits before sigmoid
        """

        # ── 1. Run DreaMS encoder ──
        # Forward pass: (B, 60, 2) → (B, 60, d_model)
        # Each of the 60 peak positions gets a 1024-dim embedding.
        sequence_output = self.encoder(spec_mask, charge)  # (B, 60, 1024)

        # ── 2. Masked Peak Prediction Loss ──
        # Reuse DreaMS's pre-trained heads (ff_out, ff_out_intens) and loss (mz_masking_loss).
        # This avoids duplicating the complex hot-encoding logic.
        masked_peak_loss = self._compute_masked_peak_loss(
            sequence_output, spec_real, mask
        )

        # ── 3. MACCS Classification Loss ──
        # Aggregate (B, 60, 1024) → (B, feat_dim) according to pooling strategy.
        # Gradient flows to ALL 60 positions, not just [CLS].
        pooled = self._pool_sequence(sequence_output)  # (B, 1024) or (B, 2048)
        maccs_logits = self.maccs_head(pooled)          # (B, 167)
        maccs_loss = F.binary_cross_entropy_with_logits(
            maccs_logits, maccs_labels.float()
        )                                             # scalar

        # ── 4. Joint Loss ──
        total_loss = 0.2 * masked_peak_loss + 1.0 * maccs_loss

        return {
            'total_loss': total_loss,
            'masked_peak_loss': masked_peak_loss.detach(),
            'maccs_loss': maccs_loss.detach(),
            'sequence_output': sequence_output,  # (B, 60, 1024) — full, no pooling
            'maccs_logits': maccs_logits,
        }

    def _compute_masked_peak_loss(
        self,
        sequence_output: torch.Tensor,
        spec_real: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Replicate DreaMS's spec_ssl_step logic without a second forward pass.

        DreaMS was pre-trained to predict masked peaks' m/z (and intensity)
        from their context. We reuse its trained heads (ff_out, ff_out_intens)
        and loss function (mz_masking_loss / FocalLoss).

        Args:
            sequence_output: (B, 60, 1024) DreaMS encoder output.
            spec_real:       (B, 60, 2) original spectrum.
            mask:            (B, 60) boolean, True = masked positions.

        Returns:
            scalar loss.
        """
        obj = self.encoder.train_objective

        if not obj.startswith('mask'):
            return torch.tensor(0.0, device=sequence_output.device)

        # Gather predictions at masked positions only
        # mask indexing: (B, 60) → (num_masked, 1024)
        pred_embs = sequence_output[mask]           # (M, 1024)
        real = spec_real[mask]                      # (M, 2) = (m/z, intensity)

        if obj.endswith('hot'):
            # ── m/z bin prediction ──
            # ff_out: (d_model) → (num_mz_bins), e.g. 1024 → 2000+ bins
            pred_mz = self.encoder.ff_out(pred_embs)  # (M, num_mz_bins)

            # Convert ground-truth m/z values to one-hot classes
            real_mz = to_hot(
                real[..., [0]],                        # (M, 1)
                max_val=self.encoder.dformat.max_mz,
                bin_size=self.encoder.hot_mz_bin_size,
            )                                          # (M, num_mz_bins)

            # Focal loss for m/z prediction
            loss, p_mz = self.encoder.mz_masking_loss(pred_mz, real_mz)
            # loss shape: (M,) — per-masked-peak loss

            # ── Intensity bin prediction (if pre-trained with it) ──
            if obj == 'mask_peak_hot' and hasattr(self.encoder, 'ff_out_intens'):
                pred_intens = self.encoder.ff_out_intens(pred_embs)  # (M, num_intens_bins)
                real_intens = to_hot(
                    real[..., [1]],                    # (M, 1)
                    max_val=1.0,
                    bin_size=0.05,
                )                                      # (M, num_intens_bins)
                loss += 0.5 * F.cross_entropy(
                    pred_intens, real_intens, reduction='none'
                )

            # Entropy label smoothing (if used in pre-training)
            if self.encoder.entropy_label_smoothing > 0:
                loss -= self.encoder.entropy_label_smoothing * \
                    torch.distributions.Categorical(p_mz).entropy()

            # Mean over masked positions → scalar
            return loss.mean()

        elif obj == 'mask_mz':
            # Continuous m/z regression
            pred_mz = self.encoder.ff_out(pred_embs).squeeze(-1)  # (M,)
            real_mz = real[..., 0]                                # (M,)
            loss = self.encoder.mz_masking_loss(pred_mz, real_mz)  # → (M,) MSE
            return loss.mean()

        else:
            raise NotImplementedError(
                f"train_objective '{obj}' not implemented in ChemFinetuner"
            )
