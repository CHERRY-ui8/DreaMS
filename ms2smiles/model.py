"""MStoSMILES: DreaMS (frozen) → Projector → ChemGPT (fine-tuned) for SMILES generation.

Architecture:
    MS embedding (1024-d) → Projector MLP (1024 → hidden_size) → V_ms (B, 1, hidden_size)
        → cat([V_ms, token_embs]) → GPTNeo (prefix-tuned) → logits

Training modes:
    1. Precomputed embeddings (default): use pairs_with_embs.hdf5, only Projector + ChemGPT
    2. End-to-end: use raw spectra, include DreaMS forward pass (slower)
"""

import json
import sys
import types
import warnings
from pathlib import Path

import torch
import torch.nn as nn

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


def build_projector(dreams_dim: int, decoder_dim: int,
                    hidden_dims: list = None) -> nn.Module:
    """Build MLP projector."""
    if hidden_dims is None:
        return nn.Linear(dreams_dim, decoder_dim)
    layers = []
    in_dim = dreams_dim
    for h in hidden_dims:
        layers.extend([nn.Linear(in_dim, h), nn.GELU()])
        in_dim = h
    layers.append(nn.Linear(in_dim, decoder_dim))
    return nn.Sequential(*layers)


# ═══════════════════════════════════════════════════════════════
# Main model
# ═══════════════════════════════════════════════════════════════
class MStoSMILES(nn.Module):
    """MS embedding → Projector → ChemGPT (prefix-tuned).

    Two input modes:
        - 'embedding' mode: accepts precomputed (B, 1024) embeddings (fast)
        - 'spectrum' mode: accepts raw (B, 2, 60) spectra (uses DreaMS internally)
    """

    def __init__(self, config: MS2SMILESConfig, device: str = 'cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self._dreams = None  # Lazy load for e2e mode
        self._tokenizer = None
        warnings.filterwarnings('ignore')

        # 1. Projector
        self.projector = build_projector(
            config.dreams_d_model, config.decoder_hidden_size,
            config.projector_hidden,
        ).to(device)
        print(f'[MStoSMILES] Projector: {sum(p.numel() for p in self.projector.parameters()):,} params')

        # 2. ChemGPT decoder
        print(f'[MStoSMILES] Loading ChemGPT ({config.model_size})...')
        self.chemgpt, self._tokenizer = load_chemgpt(config, device=device)
        n_chemgpt = sum(p.numel() for p in self.chemgpt.parameters())
        print(f'  ChemGPT: {n_chemgpt:,} params ({config.decoder_hidden_size}-dim)')
        print(f'  Total trainable: {sum(p.numel() for p in self.parameters()):,}')

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

    def forward(
        self,
        smiles_ids: torch.Tensor,
        labels: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        embeddings: torch.Tensor = None,   # (B, 1024) precomputed
        spectra: torch.Tensor = None,      # (B, 2, 60) raw — only if embeddings is None
    ):
        """Forward pass with prefix tuning.

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

        # 2. Project to ChemGPT's hidden space
        v_ms = self.projector(ms_emb).unsqueeze(1)  # (B, 1, H)

        # 3. Token embeddings
        token_embs = self.chemgpt.transformer.wte(smiles_ids)  # (B, S, H)

        # 4. Concatenate: [V_ms, tokens...]
        inputs_embeds = torch.cat([v_ms, token_embs], dim=1)  # (B, S+1, H)

        # 5. Labels: prefix position is ignored
        if labels is not None:
            prefix_labels = torch.full(
                (labels.size(0), 1), -100,
                dtype=labels.dtype, device=labels.device,
            )
            labels = torch.cat([prefix_labels, labels], dim=1)

        # 6. Attention mask: prefix always attends
        if attention_mask is not None:
            prefix_mask = torch.ones(
                (attention_mask.size(0), 1),
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

        return {'loss': outputs.loss, 'logits': outputs.logits}

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

        v_ms = self.projector(ms_emb).unsqueeze(1)
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
            )
        else:
            outputs = self.chemgpt.generate(
                inputs_embeds=v_ms,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = []
        for i in range(batch_size):
            idx = i * num_beams if num_beams > 1 else i
            tokens = outputs[idx].cpu().tolist()
            tokens = [t for t in tokens if t not in (0, 1, 2, 3, 4)]
            smi = self.tokenizer.decode(tokens)
            generated.append(smi)

        return generated
