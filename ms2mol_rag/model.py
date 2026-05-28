"""RAG for MS → SMILES: retrieve similar molecules, then generate target.

Method:
    1. DreaMS embedding → KNN search → top-3 similar molecules (SMILES)
    2. Encoder input: [MS_prefix_tokens, ctx1_tokens, ctx2_tokens, ctx3_tokens]
    3. Decoder generates: target SMILES (via cross-attention to encoder)

Anti-Lazy-Copying mechanisms (selectable at training time):
    - 'unlikelihood': penalizes model when it assigns high prob to context tokens
    - 'context_dropout': randomly drops context to force reliance on MS signal
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.neighbors import NearestNeighbors
import numpy as np


# T5 vocab constants
PAD_ID = 0
EOS_ID = 1


def load_t5_for_rag(device: str = 'cuda'):
    """Load T5-small with mirror."""
    import os
    os.environ['TRANSFORMERS_CACHE'] = '/tmp/t5cache'
    tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    model = model.to(device)
    model.train()
    return model, tokenizer


class RAGIndex:
    """KNN index for DreaMS embeddings → similar SMILES retrieval."""

    def __init__(self, k_neighbors: int = 3):
        self.k = k_neighbors
        self.index = None
        self.smiles_list = None

    def build(self, embeddings: np.ndarray, smiles_list: list[str]):
        self.smiles_list = smiles_list
        self.index = NearestNeighbors(
            n_neighbors=min(self.k + 1, len(embeddings)),
            metric='cosine', algorithm='brute',
        )
        self.index.fit(embeddings)
        print(f'[RAGIndex] Built index on {len(embeddings)} embeddings, k={self.k}')

    def search(self, query: torch.Tensor, exclude_self: bool = True,
               query_idx: int = None) -> tuple[list[str], list[float]]:
        if query.dim() == 1:
            query = query.unsqueeze(0)
        query_np = query.cpu().numpy()
        distances, indices = self.index.kneighbors(query_np)
        indices = indices[0]
        distances = distances[0]

        if exclude_self and query_idx is not None:
            self_pos = np.where(indices == query_idx)[0]
            if len(self_pos) > 0:
                self_pos = self_pos[0]
                indices = np.delete(indices, self_pos)
                distances = np.delete(distances, self_pos)

        k = min(self.k, len(indices))
        indices = indices[:k]
        distances = distances[:k]
        similar_smiles = [self.smiles_list[i] for i in indices]
        return similar_smiles, distances.tolist()

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'embeddings': self.index._fit_X, 'smiles': self.smiles_list,
                        'k': self.k, 'metric': 'cosine'}, f)
        print(f'[RAGIndex] Saved to {path}')

    def load(self, path: str):
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.build(data['embeddings'], data['smiles'])
        print(f'[RAGIndex] Loaded from {path}')


class MSToSMILES_RAG(nn.Module):
    """RAG + T5 for MS → SMILES generation with anti-lazy-copying.

    Args:
        k_tokens: MS prefix tokens (default: 16)
        k_contexts: number of retrieved similar molecules (default: 3)
        lazy_penalty: 'none', 'unlikelihood', or 'context_dropout'
        ul_alpha: weight for unlikelihood loss term (default: 0.1)
        cd_prob: probability of dropping context (default: 0.3)
        max_context_len: max tokens per context SMILES (default: 128)
    """

    def __init__(
        self,
        k_tokens: int = 16,
        k_contexts: int = 3,
        lazy_penalty: str = 'none',
        ul_alpha: float = 0.1,
        cd_prob: float = 0.3,
        use_instruction: bool = False,
        max_context_len: int = 128,
        dreams_dim: int = 1024,
        d_model: int = 512,
        device: str = 'cuda',
    ):
        super().__init__()
        self.k_tokens = k_tokens
        self.k_contexts = k_contexts
        self.lazy_penalty = lazy_penalty
        self.ul_alpha = ul_alpha
        self.cd_prob = cd_prob
        self.use_instruction = use_instruction
        self.max_context_len = max_context_len
        self.d_model = d_model
        self.device = device

        # 1. Load T5-small
        self.t5, self.tokenizer = load_t5_for_rag(device=device)

        # 2. Pre-tokenize instruction prompt (for Phase 2 RAG)
        self.instruction_tokens = None
        if use_instruction:
            instruction_text = (
                "Reference molecules share local structural fragments, "
                "not global scaffolds. Extract common substructures and "
                "combine them to build the target:"
            )
            self.instruction_tokens = torch.tensor(
                self.tokenizer(
                    instruction_text, add_special_tokens=False,
                )['input_ids'],
                device=device,
            )  # (L,)

        # 3. MS Projector
        self.projector = nn.Sequential(
            nn.Linear(dreams_dim, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * k_tokens),
        )

        # 3. RAG index (set externally)
        self.index = None

        n_proj = sum(p.numel() for p in self.projector.parameters())
        n_t5 = sum(p.numel() for p in self.t5.parameters())
        print(f'[MSToSMILES_RAG] T5-small: {n_t5:,} params')
        print(f'[MSToSMILES_RAG] Projector: {n_proj:,} params')
        print(f'[MSToSMILES_RAG] K={k_tokens} MS tokens + {k_contexts} context molecules')
        print(f'[MSToSMILES_RAG] Lazy penalty: {lazy_penalty}')
        if use_instruction:
            print(f'[MSToSMILES_RAG] Instruction prompt: ENABLED ({len(self.instruction_tokens)} tokens)')

    def set_index(self, index: RAGIndex):
        self.index = index

    def _prepare_encoder_input(
        self, ms_emb: torch.Tensor,
        context_smiles: list[list[str]] = None,
    ):
        """Build encoder inputs_embeds and return context token IDs.

        Returns:
            inputs_embeds: (B, total_tokens, d_model)
            attention_mask: (B, total_tokens)
            context_token_ids: (B, ctx_len) raw token IDs of context, for unlikelihood loss
        """
        B = ms_emb.size(0)
        device = ms_emb.device

        # MS prefix tokens
        ms_prefix = self.projector(ms_emb).view(B, self.k_tokens, self.d_model)

        context_token_ids = None

        if context_smiles is not None:
            # Context Dropout: randomly drop context to force MS reliance
            if self.lazy_penalty == 'context_dropout' and self.training:
                dropped_contexts = []
                for batch_i in range(B):
                    if random.random() < self.cd_prob:
                        dropped_contexts.append([''] * len(context_smiles[batch_i]))
                    else:
                        dropped_contexts.append(context_smiles[batch_i])
                context_smiles = dropped_contexts

            # ── Instruction prompt for Phase 2 RAG ──
            instr_embeds = None
            instr_mask = None
            if self.use_instruction and self.instruction_tokens is not None:
                instr_embeds = self.t5.shared(self.instruction_tokens)  # (L, d_model)
                instr_embeds = instr_embeds.unsqueeze(0).expand(B, -1, -1)  # (B, L, d_model)
                instr_mask = torch.ones(B, len(self.instruction_tokens), dtype=torch.long, device=device)

            # Tokenize context SMILES
            ctx_token_ids = []
            for batch_i in range(B):
                batch_tokens = []
                for smi in context_smiles[batch_i]:
                    toks = self.tokenizer(
                        smi, add_special_tokens=False,
                        max_length=self.max_context_len,
                        truncation=True,
                    )['input_ids']
                    batch_tokens.extend(toks)
                ctx_token_ids.append(torch.tensor(batch_tokens, device=device))

            # Save context token IDs for unlikelihood loss
            if self.lazy_penalty == 'unlikelihood' and self.training:
                context_token_ids = ctx_token_ids

            # Look up embeddings
            ctx_embs = []
            for batch_i in range(B):
                if len(ctx_token_ids[batch_i]) > 0:
                    ctx_embs.append(self.t5.shared(ctx_token_ids[batch_i]))
                else:
                    ctx_embs.append(torch.zeros(0, self.d_model, device=device))

            max_ctx = max(e.size(0) for e in ctx_embs)
            if max_ctx == 0:
                max_ctx = 1

            ctx_padded = torch.zeros(B, max_ctx, self.d_model, device=device)
            ctx_mask = torch.zeros(B, max_ctx, dtype=torch.long, device=device)
            for i, e in enumerate(ctx_embs):
                if e.size(0) > 0:
                    ctx_padded[i, :e.size(0)] = e
                    ctx_mask[i, :e.size(0)] = 1
            # Concatenate: [MS_prefix, instruction, ctx_tokens]
            parts = [ms_prefix]
            masks = [torch.ones(B, self.k_tokens, dtype=torch.long, device=device)]

            if instr_embeds is not None:
                parts.append(instr_embeds)
                masks.append(instr_mask)

            parts.append(ctx_padded)
            masks.append(ctx_mask)

            inputs_embeds = torch.cat(parts, dim=1)
            attention_mask = torch.cat(masks, dim=1)
        else:
            inputs_embeds = ms_prefix
            attention_mask = torch.ones(B, self.k_tokens, dtype=torch.long, device=device)

        return inputs_embeds, attention_mask, context_token_ids

    def forward(
        self,
        ms_emb: torch.Tensor,
        labels: torch.Tensor = None,
        context_smiles: list[list[str]] = None,
    ):
        """Forward pass with optional unlikelihood anti-lazy-copying loss.

        Args:
            ms_emb: (B, 1024)
            labels: (B, S) target SMILES token IDs
            context_smiles: B lists of k SMILES strings
        Returns:
            dict with loss, logits
        """
        inputs_embeds, attn_mask, ctx_token_ids = self._prepare_encoder_input(
            ms_emb, context_smiles,
        )

        outputs = self.t5(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=labels,
            return_dict=True,
        )

        loss = outputs.loss  # standard CE loss

        # ── Unlikelihood penalty: penalize high prob on context tokens ──
        if self.lazy_penalty == 'unlikelihood' and self.training and ctx_token_ids is not None:
            logits = outputs.logits  # (B, S, V)
            # T5 internally shifts labels, so we need shifted logits too
            # logits[:, :-1] predict labels[:, 1:]
            # labels at positions 0..S-2 have targets, position S-1 has no target
            shift_logits = logits[:, :-1, :].contiguous()  # (B, S-1, V)
            shift_labels = labels[:, 1:].contiguous()      # (B, S-1)

            # Build context_ids aligned with labels length
            B, S = shift_labels.shape
            ctx_ids_aligned = torch.full_like(shift_labels, -100)

            for batch_i in range(B):
                ctx_tokens = ctx_token_ids[batch_i]  # 1-D tensor
                if len(ctx_tokens) > 0:
                    ctx_len = min(len(ctx_tokens), S)
                    ctx_ids_aligned[batch_i, :ctx_len] = ctx_tokens[:ctx_len]

            # Compute context token probabilities at each position
            probs = F.softmax(shift_logits, dim=-1)  # (B, S, V)

            # Gather probability of the actual context token at each position
            ctx_probs = probs.gather(
                2, ctx_ids_aligned.unsqueeze(-1).clamp(min=0),
            ).squeeze(-1)  # (B, S)
            ctx_probs[ctx_ids_aligned == -100] = 0.0

            # Positions where label != context token (model is "copying wrong")
            valid = (shift_labels != -100) & (ctx_ids_aligned != -100)
            diff = (shift_labels != ctx_ids_aligned) & valid

            # Unlikelihood: -log(1 - P(context_token))
            penalty = -torch.log(1.0 - ctx_probs + 1e-8)

            loss_ul = (penalty * diff.float()).sum() / diff.float().sum().clamp(min=1)
            loss = loss + self.ul_alpha * loss_ul

        return {'loss': loss, 'logits': outputs.logits}

    @torch.no_grad()
    def generate(
        self,
        ms_emb: torch.Tensor,
        context_smiles: list[list[str]] = None,
        max_length: int = 200,
        num_beams: int = 5,
        device: str = 'cuda',
    ) -> list[str]:
        if ms_emb.dim() == 1:
            ms_emb = ms_emb.unsqueeze(0)
        B = ms_emb.size(0)

        inputs_embeds, attn_mask, _ = self._prepare_encoder_input(ms_emb, context_smiles)

        outputs = self.t5.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_length,
            num_beams=num_beams,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=PAD_ID,
            eos_token_id=EOS_ID,
        )

        generated = []
        for i in range(B):
            smi = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            generated.append(smi)
        return generated
