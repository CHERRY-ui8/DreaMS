"""Shared LoRA implementation for all ms2mol architectures.

Provides LoRALinear wrapper and injection functions for both GPT-Neo and T5.
"""

import torch
import torch.nn as nn


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

        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling


def inject_lora_gptneo(model: nn.Module, rank: int = 8, alpha: float = 16.0,
                       target_modules: list = None):
    """Inject LoRA into GPT-Neo attention layers (q_proj, v_proj)."""
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj']

    replacements = []
    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
            replacements.append(name)

    for name in replacements:
        parts = name.split('.')
        child_name = parts[-1]
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        lora_linear = LoRALinear(getattr(parent, child_name), rank=rank, alpha=alpha)
        lora_linear = lora_linear.to(next(parent.parameters()).device)
        setattr(parent, child_name, lora_linear)

    lora_params = sum(p.numel() for n, p in model.named_parameters()
                      if 'lora_A' in n or 'lora_B' in n)
    print(f'[LoRA] Injected {len(replacements)} GPT-Neo modules (rank={rank})')
    print(f'[LoRA] Trainable params: {lora_params:,}')
    return model


def inject_lora_t5(model: nn.Module, rank: int = 8, alpha: float = 16.0,
                   target_modules: list = None):
    """Inject LoRA into T5 attention layers.

    Typical targets for T5:
        - 'SelfAttention.q', 'SelfAttention.v' (self-attention)
        - 'EncDecAttention.q', 'EncDecAttention.v' (cross-attention, decoder only)
    """
    if target_modules is None:
        target_modules = ['SelfAttention.q', 'SelfAttention.v',
                          'EncDecAttention.q', 'EncDecAttention.v']

    replacements = []
    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
            replacements.append(name)

    for name in replacements:
        parts = name.split('.')
        child_name = parts[-1]
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        lora_linear = LoRALinear(getattr(parent, child_name), rank=rank, alpha=alpha)
        lora_linear = lora_linear.to(next(parent.parameters()).device)
        setattr(parent, child_name, lora_linear)

    lora_params = sum(p.numel() for n, p in model.named_parameters()
                      if 'lora_A' in n or 'lora_B' in n)
    print(f'[LoRA] Injected {len(replacements)} T5 modules (rank={rank})')
    print(f'[LoRA] Trainable params: {lora_params:,}')
    return model


def count_lora_params(model: nn.Module) -> int:
    return sum(p.numel() for n, p in model.named_parameters()
               if 'lora_A' in n or 'lora_B' in n)
