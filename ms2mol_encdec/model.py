"""Multi-backbone encoder-decoder for MS -> SMILES generation.

Supported backbones (configured via --model_name):
    t5-small        T5-small (60M,  d_model=512,   6+6 layers)
    t5-base         T5-base  (220M, d_model=768,  12+12 layers)
    molt5-small     MolT5-small (same as T5-small, chemical pretrain, d_model=512)
    biot5-base      BioT5-base (T5-base + biomedical pretrain, d_model=768)
    biot5-plus-base BioT5+ base (extended vocab, d_model=768)

Architecture:
    DreaMS embedding (1024-d) -> Projector (1024 -> d_model*K) -> K encoder tokens
        -> T5-family encoder (self-attn)
        -> T5-family decoder (self-attn + cross-attn to encoder) -> SMILES tokens

Cross-attention at every decoder step ensures the MS signal never gets diluted.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ms2mol_shared.lora import inject_lora_t5, count_lora_params


# ── Model registry ──────────────────────────────────────────────
# Maps CLI-friendly names to HuggingFace model IDs and metadata.
MODEL_REGISTRY = {
    't5-small': {
        'hf_name': 't5-small',
        'family': 't5',
        'display': 'T5-small',
        'default_d_model': 512,
    },
    't5-base': {
        'hf_name': 't5-base',
        'family': 't5',
        'display': 'T5-base',
        'default_d_model': 768,
    },
    'molt5-small': {
        'hf_name': 'laituan245/molt5-small',
        'family': 't5',
        'display': 'MolT5-small',
        'default_d_model': 512,
    },
    'biot5-base': {
        'hf_name': 'QizhiPei/biot5-base',
        'family': 't5',
        'display': 'BioT5-base',
        'default_d_model': 768,
    },
    'biot5-plus-base': {
        'hf_name': 'QizhiPei/biot5-plus-base',
        'family': 't5',
        'display': 'BioT5+ base',
        'default_d_model': 768,
    },
}

# All models are T5-based and share the same vocab IDs
PAD_ID = 0
EOS_ID = 1
BOS_ID = 0  # T5 uses pad_id as both pad and start-of-sequence

# ── MACCS keys ───────────────────────────────────────────────────
MACCS_NBITS = 167


class MaccsHead(nn.Module):
    """Substructure classification head: pooled projector output -> 167 MACCS keys.

    Takes the projector output (B, K, d_model), mean-pools over the K dimension,
    then applies a linear layer to predict 167 binary MACCS key probabilities.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, MACCS_NBITS)

    def forward(self, projector_out: torch.Tensor) -> torch.Tensor:
        """Returns (B, 167) logits for BCEWithLogitsLoss."""
        pooled = projector_out.mean(dim=1)  # (B, d_model)
        return self.classifier(pooled)  # (B, 167)


class KHeadsProjector(nn.Module):
    """D1 projector: shared trunk + K independent low-rank heads.

    Each of the K encoder prefix tokens gets its own small MLP (trunk_dim -> head_rank
    -> d_model), so capacity scales as O(K) instead of O(K * d_model) in one matrix.
    """

    def __init__(
        self,
        dreams_dim: int,
        d_model: int,
        k_tokens: int,
        trunk_dim: int = 512,
        head_rank: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.k_tokens = k_tokens
        self.d_model = d_model
        self.trunk_dim = trunk_dim
        self.head_rank = head_rank

        trunk_layers = [nn.Linear(dreams_dim, trunk_dim), nn.GELU()]
        if dropout > 0:
            trunk_layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*trunk_layers)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_dim, head_rank),
                nn.GELU(),
                nn.Linear(head_rank, d_model),
            )
            for _ in range(k_tokens)
        ])

    def forward(self, ms_emb: torch.Tensor) -> torch.Tensor:
        """Returns (B, K, d_model) prefix token embeddings."""
        h = self.trunk(ms_emb)
        return torch.stack([head(h) for head in self.heads], dim=1)


class QFormerProjector(nn.Module):
    """Q-Former projector: learnable queries cross-attend to MS embedding.

    Replaces MLP projector with a small transformer that uses learnable queries
    to extract information from the DreaMS embedding via cross-attention.

    Architecture:
        MS emb (1024) -> Linear -> d_model (vision token, seq_len=1)
        Learnable queries (num_queries, d_model)
        -> N × QFormerBlock(SelfAttn → CrossAttn(vision) → FFN)
        -> (B, num_queries, d_model) prefix tokens

    Compatible with MaccsHead (mean-pool over queries -> 167 logits).
    Also compatible with Phase 2 LoRA (queries + cross-attn weights trainable).
    """

    def __init__(self, dreams_dim: int, d_model: int, num_queries: int = 32,
                 num_layers: int = 4, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model

        # Project MS embedding to d_model as vision token
        self.ms_proj = nn.Linear(dreams_dim, d_model)
        self.ms_norm = nn.LayerNorm(d_model)

        # Learnable queries + position embeddings
        self.query_emb = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)

        # Q-Former transformer blocks
        self.blocks = nn.ModuleList([
            QFormerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ms_emb: torch.Tensor) -> torch.Tensor:
        """Map MS embedding to (B, num_queries, d_model) prefix tokens."""
        B = ms_emb.size(0)

        # Vision token: (B, 1, d_model)
        vis = self.ms_norm(self.ms_proj(ms_emb)).unsqueeze(1)

        # Queries: (B, num_queries, d_model)
        queries = self.query_emb.expand(B, -1, -1) + self.query_pos.expand(B, -1, -1)

        for block in self.blocks:
            queries = block(queries, vis)

        return self.norm(queries)


class QFormerBlock(nn.Module):
    """Single Q-Former transformer block.

    Order: SelfAttn(LN) + residual → CrossAttn(LN, vision) + residual → FFN(LN) + residual
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                                batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                                 batch_first=True)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        # Self-attention among queries
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q, need_weights=False)[0]

        # Cross-attention: queries attend to vision token
        q = self.norm2(queries)
        queries = queries + self.cross_attn(q, vision, vision, need_weights=False)[0]

        # FFN
        q = self.norm3(queries)
        queries = queries + self.ffn(q)

        return queries


def load_backbone(model_name: str = 't5-small', device: str = 'cuda'):
    """Load a backbone model + tokenizer with HF mirror support.

    Args:
        model_name: Key into MODEL_REGISTRY (e.g. 't5-small', 'molt5-small').
        device: Target device.

    Returns:
        (model, tokenizer, d_model)
    """
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/t5cache'

    info = MODEL_REGISTRY[model_name]
    hf_name = info['hf_name']

    print(f'[load_backbone] Loading {info["display"]} ({hf_name}) ...')
    tokenizer = AutoTokenizer.from_pretrained(hf_name, legacy=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)

    d_model = model.config.d_model  # auto-detect: 512 for small, 768 for base
    print(f'[load_backbone]   d_model={d_model}, vocab_size={model.config.vocab_size}')

    model = model.to(device)
    model.train()
    return model, tokenizer, d_model


class MSToSMILES_T5(nn.Module):
    """MS embedding -> Projector -> T5-family encoder -> T5-family decoder -> SMILES.

    The projector maps the DreaMS embedding to K tokens in the backbone's
    embedding space. These tokens are fed to the encoder, and the decoder
    generates SMILES with cross-attention to the encoder output at every step.

    Args:
        k_tokens: Number of prefix tokens (default: 16).
        dreams_dim: DreaMS embedding dimension (default: 1024).
        model_name: Backbone model name (key into MODEL_REGISTRY).
        device: Target device.
    """

    def __init__(self, k_tokens: int = 16, dreams_dim: int = 1024,
                 model_name: str = 't5-small', device: str = 'cuda',
                 projector_type: str = 'mlp',
                 projector_depth: int = 2, projector_dropout: float = 0.0,
                 projector_trunk_dim: int = 512, projector_head_rank: int = 64,
                 qformer_num_queries: int = 32, qformer_layers: int = 4,
                 qformer_heads: int = 8,
                 ms_decoder_adapter: bool = False,
                 maccs_loss_weight: float = 0.0,
                 ce_loss_weight: float = 1.0):
        super().__init__()
        self.k_tokens = k_tokens
        self.model_name = model_name
        self.device = device
        self.projector_type = projector_type
        self.projector_depth = projector_depth
        self.projector_dropout = projector_dropout
        self.projector_trunk_dim = projector_trunk_dim
        self.projector_head_rank = projector_head_rank
        self.qformer_num_queries = qformer_num_queries
        self.qformer_layers = qformer_layers
        self.qformer_heads = qformer_heads
        self.ms_decoder_adapter = ms_decoder_adapter
        self.maccs_loss_weight = maccs_loss_weight
        self.ce_loss_weight = ce_loss_weight

        # 1. Load backbone (T5 / MolT5 / BioT5)
        self.t5, self.tokenizer, self.d_model = load_backbone(
            model_name=model_name, device=device,
        )
        self.vocab_size = self.t5.config.vocab_size

        # 2. MS Projector: dreams_dim -> prefix tokens
        # For Q-Former, num_queries replaces k_tokens as the prefix count
        if projector_type == 'qformer':
            self.k_tokens = qformer_num_queries
        self.projector = self._build_projector(dreams_dim)

        # 3. Shared embedding (for token-to-embed lookup)
        self.shared = self.t5.shared

        # 4. MACCS substructure head (optional)
        if maccs_loss_weight > 0:
            self.maccs_head = MaccsHead(d_model=self.d_model)
            print(f'[MSToSMILES_T5] MACCS substructure head: {self.maccs_head.classifier.weight.shape} '
                  f'({sum(p.numel() for p in self.maccs_head.parameters()):,} params)')
            print(f'[MSToSMILES_T5]   Loss weights: ce={ce_loss_weight}, maccs_bce={maccs_loss_weight}')
        else:
            self.maccs_head = None

        # 4. Optional MS decoder adapter: project MS to d_model and append
        #    to encoder output as an extra token for decoder cross-attention.
        #    This gives the decoder a direct line to the raw MS embedding,
        #    bypassing the projector bottleneck.
        if ms_decoder_adapter:
            self.ms_proj = nn.Linear(dreams_dim, self.d_model)
            total_adapter = sum(p.numel() for p in self.ms_proj.parameters())
        else:
            self.ms_proj = None
            total_adapter = 0

        total_t5 = sum(p.numel() for p in self.t5.parameters())
        total_proj = sum(p.numel() for p in self.projector.parameters())
        print(f'[MSToSMILES_T5] Backbone: {info_display(model_name)}: {total_t5:,} params')
        if projector_type == 'k_heads':
            print(f'[MSToSMILES_T5] Projector (k_heads/D1): {total_proj:,} params '
                  f'(trunk {dreams_dim}->{projector_trunk_dim}, '
                  f'K={k_tokens} x ({projector_trunk_dim}->{projector_head_rank}->{self.d_model}))')
        elif projector_type == 'qformer':
            print(f'[MSToSMILES_T5] Projector (qformer): {total_proj:,} params '
                  f'({qformer_num_queries} queries, {qformer_layers} layers, '
                  f'{qformer_heads} heads, 1024->{self.d_model})')
        else:
            print(f'[MSToSMILES_T5] Projector (mlp): {total_proj:,} params '
                  f'({dreams_dim} -> {self.d_model} x {self.k_tokens}, depth={projector_depth}'
                  f', dropout={self.projector_dropout:.1f})' if self.projector_dropout > 0 else
                  f'({dreams_dim} -> {self.d_model} x {self.k_tokens}, depth={projector_depth})')
        print(f'[MSToSMILES_T5] K={self.k_tokens} encoder prefix tokens, cross-attn at all decoder layers')
        if ms_decoder_adapter:
            print(f'[MSToSMILES_T5] MS decoder adapter: {total_adapter:,} params '
                  f'(project MS {dreams_dim}->{self.d_model}, append to encoder output)')

    def _build_projector(self, dreams_dim: int = 1024) -> nn.Module:
        if self.projector_type == 'k_heads':
            return KHeadsProjector(
                dreams_dim=dreams_dim,
                d_model=self.d_model,
                k_tokens=self.k_tokens,
                trunk_dim=self.projector_trunk_dim,
                head_rank=self.projector_head_rank,
                dropout=self.projector_dropout,
            )
        if self.projector_type == 'qformer':
            return QFormerProjector(
                dreams_dim=dreams_dim,
                d_model=self.d_model,
                num_queries=self.qformer_num_queries,
                num_layers=self.qformer_layers,
                num_heads=self.qformer_heads,
                dropout=0.1,
            )
        return self._build_mlp_projector(dreams_dim)

    def _encode_prefix(self, ms_emb: torch.Tensor) -> torch.Tensor:
        """Map MS embeddings to (B, K, d_model) encoder prefix tokens."""
        B = ms_emb.size(0)
        out = self.projector(ms_emb)
        if out.dim() == 2:
            return out.view(B, self.k_tokens, self.d_model)
        return out

    def _build_mlp_projector(self, dreams_dim: int = 1024) -> nn.Sequential:
        """Build a configurable-depth projector with optional dropout.

        Architecture:
            Depth=2 (original):  Linear(dreams_dim, d_model*2) → GELU → Dropout → Linear(d_model*2, d_model*K)
            Depth>=3:            Linear(dreams_dim, hidden) → GELU → Dropout →
                                   [Linear(hidden, hidden) → GELU → Dropout] × (depth-2) →
                                   Linear(hidden, d_model*K)

        Where hidden = max(d_model * 2, dreams_dim) ensuring the bottleneck is at least as
        wide as the input. Dropout only added when self.projector_dropout > 0.
        """
        hidden_dim = max(self.d_model * 2, dreams_dim)
        out_dim = self.d_model * self.k_tokens

        def maybe_dropout():
            if self.projector_dropout > 0:
                return [nn.Dropout(self.projector_dropout)]
            return []

        layers = []

        if self.projector_depth == 2:
            # Original architecture — exactly preserves backward compat
            layers = [
                nn.Linear(dreams_dim, self.d_model * 2),
                nn.GELU(),
            ]
            layers.extend(maybe_dropout())
            # Final projection might go through same or different hidden dim
            if self.d_model * 2 == out_dim:
                layers.append(nn.Identity())
            else:
                layers.append(nn.Linear(self.d_model * 2, out_dim))
            return nn.Sequential(*layers)

        # Depth >= 3: build deeper projector
        # First layer: dreams_dim → hidden_dim
        layers.append(nn.Linear(dreams_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.extend(maybe_dropout())

        # Middle layers: hidden_dim → hidden_dim
        for d in range(self.projector_depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.extend(maybe_dropout())

        # Final projection: hidden_dim → d_model * K
        layers.append(nn.Linear(hidden_dim, out_dim))

        return nn.Sequential(*layers)

    def forward(
        self,
        ms_emb: torch.Tensor,
        labels: torch.Tensor = None,
        decoder_input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        maccs: torch.Tensor = None,
    ):
        """Forward pass.

        Args:
            ms_emb: (B, 1024) DreaMS embedding.
            labels: (B, S) target token IDs (T5 handles internal shift).
            decoder_input_ids: (B, S) optional, for generation.
            attention_mask: (B, S) decoder padding mask.
            maccs: (B, 167) optional MACCS key targets for substructure supervision.

        Returns:
            dict with loss, logits, and optionally maccs_logits, maccs_loss.
        """
        # 1. Project MS to K prefix tokens
        prefix = self._encode_prefix(ms_emb)
        B = ms_emb.size(0)
        device = ms_emb.device

        # 2. Optional MACCS substructure prediction
        maccs_logits = None
        bce_loss = None
        if self.maccs_head is not None:
            maccs_logits = self.maccs_head(prefix)  # (B, 167)
            if maccs is not None:
                bce_loss = F.binary_cross_entropy_with_logits(
                    maccs_logits, maccs.float()
                )

        if self.ms_proj is not None:
            # ── MS decoder adapter path ──────────────────────────────
            # Encode prefix tokens first
            encoder_out = self.t5.encoder(
                inputs_embeds=prefix,
                attention_mask=torch.ones(B, self.k_tokens, dtype=torch.long,
                                          device=device),
                return_dict=True,
            )
            # Project MS embedding to d_model as an extra token
            ms_token = self.ms_proj(ms_emb).unsqueeze(1)  # (B, 1, d_model)
            # Append to encoder output: decoder cross-attends to both
            combined = torch.cat([encoder_out.last_hidden_state, ms_token], dim=1)
            combined_mask = torch.ones(B, self.k_tokens + 1, dtype=torch.long,
                                       device=device)

            outputs = self.t5(
                encoder_outputs=(combined,),
                attention_mask=combined_mask,
                labels=labels,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            # ── Original path ────────────────────────────────────────
            outputs = self.t5(
                inputs_embeds=prefix,
                attention_mask=torch.ones(B, self.k_tokens, dtype=torch.long,
                                          device=device),
                labels=labels,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=attention_mask,
                return_dict=True,
            )

        # Combine losses
        ce_loss = outputs.loss
        total_loss = self.ce_loss_weight * ce_loss
        if bce_loss is not None:
            total_loss = total_loss + self.maccs_loss_weight * bce_loss

        return {
            'loss': total_loss,
            'logits': outputs.logits,
            'ce_loss': ce_loss,
            'maccs_logits': maccs_logits,
            'maccs_loss': bce_loss,
        }

    @torch.no_grad()
    def generate(
        self,
        ms_emb: torch.Tensor,
        max_length: int = 200,
        num_beams: int = 5,
        device: str = 'cuda',
    ) -> list[str]:
        """Generate SMILES from MS embedding.

        Args:
            ms_emb: (B, 1024) or (1, 1024).
            max_length: max SMILES length.
            num_beams: beam search width.

        Returns:
            list of SMILES strings.
        """
        if ms_emb.dim() == 1:
            ms_emb = ms_emb.unsqueeze(0)
        B = ms_emb.size(0)
        device = ms_emb.device

        prefix = self._encode_prefix(ms_emb)

        if self.ms_proj is not None:
            # ── MS decoder adapter path ──────────────────────────────
            encoder_out = self.t5.encoder(
                inputs_embeds=prefix,
                attention_mask=torch.ones(B, self.k_tokens, dtype=torch.long,
                                          device=device),
                return_dict=True,
            )
            ms_token = self.ms_proj(ms_emb).unsqueeze(1)  # (B, 1, d_model)
            combined = torch.cat([encoder_out.last_hidden_state, ms_token], dim=1)
            combined_mask = torch.ones(B, self.k_tokens + 1, dtype=torch.long,
                                       device=device)

            gen_kwargs = dict(
                encoder_outputs=(combined,),
                attention_mask=combined_mask,
                max_new_tokens=max_length,
                pad_token_id=PAD_ID,
                eos_token_id=EOS_ID,
            )
        else:
            gen_kwargs = dict(
                inputs_embeds=prefix,
                max_new_tokens=max_length,
                pad_token_id=PAD_ID,
                eos_token_id=EOS_ID,
            )

        if num_beams > 1:
            outputs = self.t5.generate(
                **gen_kwargs,
                num_beams=num_beams,
                do_sample=False,
                num_return_sequences=1,
            )
        else:
            outputs = self.t5.generate(
                **gen_kwargs,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )

        generated = []
        for i in range(B):
            smi = self.tokenizer.decode(
                outputs[i], skip_special_tokens=True,
            )
            generated.append(smi)

        return generated


def info_display(model_name: str) -> str:
    """Return the display name for a model (or the raw name if unknown)."""
    return MODEL_REGISTRY.get(model_name, {}).get('display', model_name)
