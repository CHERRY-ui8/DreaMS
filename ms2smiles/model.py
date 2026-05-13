"""MStoSMILES: DreaMS (frozen) → Projector → ChemGPT (prefix-tuned) for SMILES generation.

Architecture:
    MS embedding (1024-d) → Projector MLP (1024 → K×hidden_size) → V_ms (B, K, hidden_size)
        → cat([V_ms_1..K, token_embs]) → GPTNeo (prefix-tuned) → logits

Fixes applied:
    - Multi-token prefix: K tokens instead of 1 (Fix 1)
    - LoRA on attention projections (Fix 4)
    - EOS token in labels (Step 0)
    - Time-step loss reweighting in training loop (Fix 2)
"""

import json
import sys
import types
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ms2smiles.config import MS2SMILESConfig


# ═══════════════════════════════════════════════════════════════
# DreaMS loader (lazy — only loaded in e2e mode)
# ═══════════════════════════════════════════════════════════════
_DREAMS_CACHE = {}


def load_dreams(ckpt_path: str, device: str = 'cpu'):
    """Load DreaMS backbone, cache globally."""
    global _DREAMS_CACHE
    key = (ckpt_path, device)
    if key in _DREAMS_CACHE:
        return _DREAMS_CACHE[key]

    # CUDA warmup BEFORE TensorFlow imports
    if device != 'cpu' and torch.cuda.is_available():
        _ = torch.zeros(1, device=device)
        del _

    import dreams.models.dreams.dreams as dm
    import dreams.models.dreams.layers as dl
    import dreams.models.layers.fourier_features as ff
    import dreams.models.layers.feed_forward as fw
    import dreams.utils.data as du
    import dreams.utils.dformats as dformats
    import dreams.utils.spectra as su

    # Mock old package namespace
    for ns in ['msml', 'msml.models', 'msml.models.dreams',
               'msml.models.layers', 'msml.utils']:
        sys.modules[ns] = types.ModuleType(ns)
    sys.modules['msml.models.dreams.dreams'] = dm
    sys.modules['msml.models.dreams.layers'] = dl
    sys.modules['msml.models.layers.fourier_features'] = ff
    sys.modules['msml.models.layers.feed_forward'] = fw
    sys.modules['msml.utils.data'] = du
    sys.modules['msml.utils.dformats'] = dformats
    sys.modules['msml.utils.spectra'] = su

    from argparse import Namespace
    from dreams.utils.data import SpectrumPreprocessor
    from dreams.utils.dformats import DataFormatA
    from dreams.models.dreams.dreams import DreaMS

    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = raw['state_dict']
    hp = raw['hyper_parameters']['args']
    if hasattr(hp, '__dict__'):
        hp = vars(hp)

    dformat = DataFormatA()
    spec_preproc = SpectrumPreprocessor(dformat=dformat, n_highest_peaks=60)

    clean = {'dformat': dformat, 'no_transformer_bias': True}
    for k, v in hp.items():
        if isinstance(v, Path):
            clean[k] = str(v)
        else:
            clean[k] = v
    clean['enable_cond_tokens'] = False

    model = DreaMS(Namespace(**clean), spec_preproc)
    load_sd = {k: v for k, v in sd.items()
               if not any(x in k for x in ['ro_out', 'mz_masking'])}
    model.load_state_dict(load_sd, strict=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    _DREAMS_CACHE[key] = (model, spec_preproc)
    return model, spec_preproc


def load_chemgpt(config: MS2SMILESConfig, device: str = 'cpu'):
    """Load ChemGPT (GPTNeoForCausalLM) from local checkpoint."""
    from transformers import (
        PreTrainedTokenizerFast,
        GPTNeoForCausalLM,
        GPTNeoConfig,
        GenerationConfig,
    )

    with open(config.chemgpt_config) as f:
        cfg_dict = json.load(f)
    model_config = GPTNeoConfig(**cfg_dict)

    gen_config = GenerationConfig(
        max_length=config.max_generate_length,
        do_sample=True,
        temperature=1.0,
    )

    model = GPTNeoForCausalLM.from_pretrained(
        config.chemgpt_ckpt,
        config=model_config,
        generation_config=gen_config,
        ignore_mismatched_sizes=True,
    )
    model = model.to(device)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.chemgpt_tokenizer)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    return model, tokenizer


def build_projector(dreams_dim: int, decoder_dim: int, k_tokens: int = 4) -> nn.Module:
    """Build MLP projector for multi-token prefix.

    Outputs K * decoder_dim, reshaped to (B, K, decoder_dim) in forward().
    Uses a hidden layer for better expressivity.
    """
    return nn.Sequential(
        nn.Linear(dreams_dim, decoder_dim * 2),
        nn.GELU(),
        nn.Linear(decoder_dim * 2, decoder_dim * k_tokens),
    )


# ═══════════════════════════════════════════════════════════════
# LoRA implementation (no peft dependency)
# ═══════════════════════════════════════════════════════════════

class LoRALinear(nn.Module):
    """Low-rank adapter wrapping an existing nn.Linear layer.

    Forward: y = Wx + (alpha / rank) * B(Ax)
    Only A and B are trainable; W is frozen.
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Freeze base weights
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        # LoRA low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0,
                target_modules: list = None):
    """Inject LoRA into specified linear layers of a GPTNeo model.

    Replaces target layers in-place with LoRALinear wrappers.
    """
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj']

    replacements = {}
    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
            # Check it's a direct child (not nested)
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            replacements[name] = (parent_name, child_name, module)

    for name, (parent_name, child_name, base_linear) in replacements.items():
        parent = model
        if parent_name:
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
        lora_linear = LoRALinear(base_linear, rank=rank, alpha=alpha)
        setattr(parent, child_name, lora_linear)
        # Ensure only LoRA params are trainable
        for p in lora_linear.lora_A.parameters():
            p.requires_grad = True
        for p in lora_linear.lora_B.parameters():
            p.requires_grad = True

    n_lora = sum(p.numel() for p in model.parameters()
                 if hasattr(p, 'requires_grad') and p.requires_grad
                 and 'lora_' in str(p.__class__.__name__).lower()
                 or any('lora' in n for n, _ in model.named_parameters()
                        if p is _))
    # Actually count LoRA params precisely
    lora_params = sum(p.numel() for n, p in model.named_parameters()
                      if 'lora_A' in n or 'lora_B' in n)
    print(f'[LoRA] Injected {len(replacements)} modules (rank={rank}, alpha={alpha})')
    print(f'[LoRA] Trainable LoRA params: {lora_params:,}')
    return model


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for n, p in model.named_parameters()
               if 'lora_A' in n or 'lora_B' in n)


# ═══════════════════════════════════════════════════════════════
# Main model
# ═══════════════════════════════════════════════════════════════
class MStoSMILES(nn.Module):
    """MS embedding → Projector → ChemGPT (multi-token prefix).

    Two input modes:
        - 'embedding' mode: accepts precomputed (B, 1024) embeddings (fast)
        - 'spectrum' mode: accepts raw (B, 2, 60) spectra (uses DreaMS internally)
    """

    def __init__(self, config: MS2SMILESConfig, device: str = 'cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.k_tokens = config.k_tokens
        self._dreams = None  # Lazy load for e2e mode
        self._tokenizer = None
        warnings.filterwarnings('ignore')

        # 1. Multi-token projector
        self.projector = build_projector(
            config.dreams_d_model, config.decoder_hidden_size,
            k_tokens=config.k_tokens,
        ).to(device)
        proj_params = sum(p.numel() for p in self.projector.parameters())
        print(f'[MStoSMILES] Projector: {proj_params:,} params '
              f'(1024 → {config.decoder_hidden_size}×{config.k_tokens})')

        # 2. ChemGPT decoder
        print(f'[MStoSMILES] Loading ChemGPT ({config.model_size})...')
        self.chemgpt, self._tokenizer = load_chemgpt(config, device=device)
        n_chemgpt = sum(p.numel() for p in self.chemgpt.parameters())
        print(f'  ChemGPT: {n_chemgpt:,} params ({config.decoder_hidden_size}-dim)')

    @property
    def tokenizer(self):
        return self._tokenizer

    def _get_dreams(self):
        """Lazy-load DreaMS (only when end-to-end mode is needed)."""
        if self._dreams is None:
            print('[MStoSMILES] Lazy-loading DreaMS backbone...')
            model, _ = load_dreams(self.config.dreams_ckpt, device=self.device)
            self._dreams = model
        return self._dreams

    def extract_ms_embedding(self, spectra: torch.Tensor) -> torch.Tensor:
        """Extract DreaMS position-0 embedding from raw spectra.

        Args:
            spectra: (B, 2, 60) — [mz, intensity]
        Returns:
            (B, 1024)
        """
        dreams = self._get_dreams()
        with torch.no_grad():
            peaks = spectra.permute(0, 2, 1).to(self.device)  # (B, 60, 2)
            embs = dreams(peaks, charge=None)
        return embs[:, 0, :]  # (B, 1024)

    def embed_ms(self, ms_emb: torch.Tensor) -> torch.Tensor:
        """Project MS embedding to multi-token prefix.

        Args:
            ms_emb: (B, 1024)
        Returns:
            (B, K, H) where K = self.k_tokens, H = hidden_size
        """
        B = ms_emb.size(0)
        flat = self.projector(ms_emb)  # (B, K * H)
        return flat.view(B, self.k_tokens, -1)  # (B, K, H)

    def forward(
        self,
        smiles_ids: torch.Tensor,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        embeddings: torch.Tensor = None,   # (B, 1024) precomputed
        spectra: torch.Tensor = None,      # (B, 2, 60) raw — only if embeddings is None
    ):
        """Forward pass with multi-token prefix tuning.

        Provide EITHER embeddings OR spectra (not both).

        Args:
            smiles_ids: (B, S) tokenized SMILES
            labels: (B, S) LM labels
            attention_mask: (B, S) padding mask
            embeddings: (B, 1024) precomputed MS embeddings
            spectra: (B, 2, 60) raw spectra
        Returns:
            dict with loss, logits
        """
        # 1. Get MS embedding
        if embeddings is not None:
            ms_emb = embeddings.to(self.device)
        elif spectra is not None:
            ms_emb = self.extract_ms_embedding(spectra)
        else:
            raise ValueError('Must provide either embeddings or spectra')

        # 2. Project to multi-token prefix (B, K, H)
        v_ms = self.embed_ms(ms_emb)

        # 3. Token embeddings
        token_embs = self.chemgpt.transformer.wte(smiles_ids)  # (B, S, H)

        # 4. Concatenate: [V_ms_1, ..., V_ms_K, tokens...]
        inputs_embeds = torch.cat([v_ms, token_embs], dim=1)  # (B, K+S, H)

        # 5. Labels: all K prefix positions are ignored
        if labels is not None:
            prefix_labels = torch.full(
                (labels.size(0), self.k_tokens), -100,
                dtype=labels.dtype, device=labels.device,
            )
            labels = torch.cat([prefix_labels, labels], dim=1)

        # 6. Attention mask: prefix always attends
        if attention_mask is not None:
            prefix_mask = torch.ones(
                (attention_mask.size(0), self.k_tokens),
                dtype=attention_mask.dtype, device=attention_mask.device,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 7. GPTNeo forward
        outputs = self.chemgpt(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

        return {'loss': outputs.loss, 'logits': outputs.logits, 'labels': labels}

    @torch.no_grad()
    def generate(
        self,
        embeddings: torch.Tensor = None,
        spectra: torch.Tensor = None,
        max_length: int = 200,
        num_beams: int = 5,
    ) -> list[str]:
        """Generate SMILES from MS embeddings or spectra."""
        if embeddings is not None:
            ms_emb = embeddings.to(self.device)
        elif spectra is not None:
            ms_emb = self.extract_ms_embedding(spectra)
        else:
            raise ValueError('Provide embeddings or spectra')
        if ms_emb.dim() == 1:
            ms_emb = ms_emb.unsqueeze(0)

        # Multi-token prefix
        v_ms = self.embed_ms(ms_emb)  # (B, K, H), no length expansion needed
        batch_size = v_ms.size(0)

        if num_beams > 1:
            v_ms_exp = v_ms.repeat_interleave(num_beams, dim=0)
            outputs = self.chemgpt.generate(
                inputs_embeds=v_ms_exp,
                max_new_tokens=max_length,
                num_beams=num_beams,
                do_sample=False,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=3,  # token [SEP] = EOS
            )
        else:
            outputs = self.chemgpt.generate(
                inputs_embeds=v_ms,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=3,  # token [SEP] = EOS
            )

        generated = []
        for i in range(batch_size):
            idx = i * num_beams if num_beams > 1 else i
            tokens = outputs[idx].cpu().tolist()
            # Strip BOS (2) at start, EOS (3) and everything after EOS at end
            # Also remove PAD=1, UNK=0, MASK=4
            tokens = [t for t in tokens if t not in (0, 1, 4)]
            if 2 in tokens:
                tokens = tokens[tokens.index(2) + 1:]
            if 3 in tokens:
                tokens = tokens[:tokens.index(3)]
            smi = self.tokenizer.decode(tokens)
            generated.append(smi)

        return generated
