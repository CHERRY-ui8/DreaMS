"""Configuration for MS→SMILES model training."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MS2SMILESConfig:
    # ── Model selection ──
    # '19M' for ChemGPT-19M (hidden_size=256, fast prototype)
    # '1.2B' for ChemGPT-1.2B (hidden_size=2048, full model)
    model_size: str = '19M'

    # ── DreaMS ──
    dreams_ckpt: str = '/root/DreaMS/dreams/models/pretrained/ssl_model.ckpt'
    dreams_d_model: int = 1024

    # ── ChemGPT paths ──
    chemgpt_19m_ckpt: str = (
        '/root/.cache/huggingface/hub/models--ncfrey--ChemGPT-19M'
        '/snapshots/7a2fecd65059d778807939848915b16efebb9cff'
    )
    chemgpt_19m_config: str = (
        '/root/.cache/huggingface/hub/models--ncfrey--ChemGPT-19M'
        '/snapshots/08876002a3a2e6f47cc454ba4153c6cffb6dd206/config.json'
    )
    chemgpt_1_2b_ckpt: str = (
        '/root/.cache/huggingface/hub/models--ncfrey--ChemGPT-1.2B'
        '/snapshots/f743bbb1e66a4864045b3db612c2fe97c0c34969'
    )
    chemgpt_1_2b_config: str = (
        '/root/.cache/huggingface/hub/models--ncfrey--ChemGPT-1.2B'
        '/snapshots/0164ca1f1754cd36b43c34b185373ee3672e7d65/config.json'
    )
    chemgpt_tokenizer: str = (
        '/root/.cache/huggingface/hub/models--ncfrey--ChemGPT-1.2B'
        '/snapshots/0164ca1f1754cd36b43c34b185373ee3672e7d65'
    )

    @property
    def decoder_hidden_size(self) -> int:
        return 256 if self.model_size == '19M' else 2048

    @property
    def chemgpt_ckpt(self) -> str:
        return self.chemgpt_19m_ckpt if self.model_size == '19M' else self.chemgpt_1_2b_ckpt

    @property
    def chemgpt_config(self) -> str:
        return self.chemgpt_19m_config if self.model_size == '19M' else self.chemgpt_1_2b_config

    # ── Multi-token prefix conditioning (Fix 1) ──
    # Number of prefix tokens to prepend (K in multi-token prefix)
    # Projector outputs K * decoder_hidden_size vectors, reshaped to (B, K, H)
    k_tokens: int = 4

    # ── LoRA (Fix 4) ──
    lora_rank: int = 8
    lora_alpha: float = 16.0
    # Which modules to apply LoRA to (comma-separated: q_proj, v_proj, k_proj, out_proj)
    lora_target_modules: str = 'q_proj,v_proj'

    # ── Training phases (Fix 3) ──
    # Phase 1 (alignment, 3-5 epochs): train ONLY projector
    # Phase 2 (lora tuning): train projector + LoRA
    # Set to '1' or '2' via command line
    phase: int = 1

    # ── Time-step Loss Reweighting (Fix 2) ──
    loss_reweight: bool = True
    # Weights for first N real-token predictions (in order: t1, t2, t3, t4, ...)
    # After padding with -100 for prefix tokens, shift_labels has:
    #   [-100, ..., -100, t1, t2, t3, t4, ...] where first real pred is at index (k_tokens-1)
    loss_reweight_values: tuple = (10.0, 8.0, 5.0, 2.0)

    # ── Training hyperparams ──
    batch_size: int = 32
    max_seq_len: int = 512
    max_epochs: int = 50
    lr_projector: float = 3e-4       # projector higher LR (random init)
    lr_chemgpt_emb: float = 3e-5     # embedding + lm_head
    lr_chemgpt_backbone: float = 1e-5  # attention layers (if unfrozen)
    lr_lora: float = 3e-4            # LoRA params (random init, higher LR)
    warmup_steps: int = 500
    grad_clip: float = 1.0
    weight_decay: float = 0.01

    # ── Data ──
    data_hdf5: str = '/root/datasets/pairs_with_embs.hdf5'
    # If using raw data without pre-extracted embeddings:
    data_raw_hdf5: str = '/root/datasets/pairs_ready.hdf5'
    use_precomputed_embs: bool = True  # False if want to compute on the fly

    # ── Inference ──
    max_generate_length: int = 200
    num_beams: int = 5
