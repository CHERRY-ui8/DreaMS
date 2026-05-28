"""Quick smoke test: load model, run 1 batch, generate 1 SMILES."""
import sys
sys.path.insert(0, '/root/DreaMS')

import torch
from torch.utils.data import DataLoader

from ms2mol_prefix.config import MS2SMILESConfig
from ms2mol_prefix.model import MStoSMILES
from ms2mol_prefix.dataset import MSSpectrumSmilesDataset, collate_fn


def freeze_strategy(model, freeze_backbone=True):
    """Apply freeze strategy for smoke test."""
    if not freeze_backbone:
        for p in model.chemgpt.parameters():
            p.requires_grad = True
        return
    for p in model.chemgpt.parameters():
        p.requires_grad = False
    for name, p in model.chemgpt.named_parameters():
        if 'lm_head' in name or name == 'transformer.wte.weight':
            p.requires_grad = True


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# Config
config = MS2SMILESConfig(model_size='19M', batch_size=4)
print(f'Model: ChemGPT-{config.model_size} (hidden_size={config.decoder_hidden_size})')

# Dataset
print('\n--- Loading dataset ---')
dataset = MSSpectrumSmilesDataset(config, split='train')
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
batch = next(iter(loader))
print(f'Batch: smiles_ids={batch["smiles_ids"].shape}, embeddings={batch["embeddings"].shape}')

# Model
print('\n--- Building model ---')
model = MStoSMILES(config, device=device)
freeze_strategy(model, freeze_backbone=True)

# Forward
print('\n--- Forward ---')
smiles_ids = batch['smiles_ids'].to(device)
labels = batch['labels'].to(device)
attention_mask = batch['attention_mask'].to(device)
embeddings = batch['embeddings'].to(device)

outputs = model(
    smiles_ids=smiles_ids,
    labels=labels,
    attention_mask=attention_mask,
    embeddings=embeddings,
)
print(f'Loss: {outputs["loss"].item():.4f}')
print(f'Logits: {outputs["logits"].shape}')

# Check gradient flow
outputs['loss'].backward()
proj_grad = model.projector.weight.grad  # first Linear in projector
lm_head_grad = model.chemgpt.lm_head.weight.grad
print(f'Projector[0] grad norm: {proj_grad.norm().item():.6f}')
print(f'LM head grad norm:     {lm_head_grad.norm().item():.6f}')
print(f'wte grad norm:         {model.chemgpt.transformer.wte.weight.grad.norm().item():.6f}')
# backbone should be 0
backbone_grad_norm = sum(
    p.grad.norm().item() for n, p in model.chemgpt.named_parameters()
    if p.grad is not None and 'lm_head' not in n and n != 'transformer.wte.weight'
)
print(f'Backbone grad norm:    {backbone_grad_norm:.6f} (should be 0 if frozen)')

# Generate
print('\n--- Generate ---')
sample_emb = embeddings[:1]
gen_smi = model.generate(embeddings=sample_emb, num_beams=3)[0]
ref_smi = dataset.smiles[0]
print(f'Reference: {ref_smi}')
print(f'Generated: {gen_smi}')

# Trainable params
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'\nTrainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')

print('\n✓ Smoke test passed!')
