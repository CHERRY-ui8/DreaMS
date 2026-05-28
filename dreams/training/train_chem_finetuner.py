"""
train_chem_finetuner.py — Continual pre-training of DreaMS with MACCS + SSL.

Data flow:
    pairs_ready.hdf5 (raw spectra) → random mask generation → ChemFinetuner
                                                                  ├── masked_peak_loss (reuse DreaMS heads)
                                                                  └── maccs_loss (new 2-layer MLP)

Usage:
    python dreams/training/train_chem_finetuner.py \
        --dataset /root/datasets/pairs_ready.hdf5 \
        --checkpoint /root/DreaMS/dreams/models/pretrained/ssl_model.ckpt \
        --max_epochs 10 \
        --batch_size 64
"""

import os, sys, time, warnings, types
from pathlib import Path
warnings.filterwarnings('ignore')
sys.path.insert(0, '/root/DreaMS')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# ── DreaMS module mock (for checkpoint loading) ──
import dreams.utils.data as du
import dreams.utils.dformats as dformats
import dreams.utils.spectra as su
import dreams.models.dreams.dreams as dm
import dreams.models.dreams.layers as dl
import dreams.models.layers.fourier_features as ff
import dreams.models.layers.feed_forward as fw
from argparse import Namespace
for ns in ['msml', 'msml.models', 'msml.models.dreams', 'msml.models.layers', 'msml.utils']:
    sys.modules[ns] = types.ModuleType(ns)
sys.modules['msml.models.dreams.dreams'] = dm
sys.modules['msml.models.dreams.layers'] = dl
sys.modules['msml.models.layers.fourier_features'] = ff
sys.modules['msml.models.layers.feed_forward'] = fw
sys.modules['msml.utils.data'] = du
sys.modules['msml.utils.dformats'] = dformats
sys.modules['msml.utils.spectra'] = su

from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from dreams.models.dreams.dreams import DreaMS
from dreams.models.dreams.chem_finetuner import DreaMS_ChemFinetuner

# ── RDKit for MACCS ──
from rdkit import Chem
from rdkit.Chem import MACCSkeys

# ── Constants ──
N_PEAKS = 60
D_MODEL = 1024
MACCS_NBITS = 167
MASK_FRAC = 0.15      # fraction of peaks to mask
MASK_VAL = -1.0        # mask value (DreaMS convention)


def smi_to_maccs(smiles: str) -> torch.Tensor:
    """SMILES → 167-bit MACCS fingerprint tensor."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return torch.zeros(MACCS_NBITS, dtype=torch.float)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return torch.tensor([fp.GetBit(i) for i in range(MACCS_NBITS)], dtype=torch.float)


class SpectraMaccsDataset(Dataset):
    """Load raw spectra + SMILES from pairs_ready.hdf5, generate masks + MACCS on-the-fly.

    Each item:
        spec_mask:  (60, 2) masked spectrum  (masked peaks → 0)
        spec_real:  (60, 2) original spectrum
        mask:       (60,)   boolean, True = masked positions
        maccs:      (167,)  MACCS fingerprint
    """

    def __init__(self, hdf5_path: str, split: str = 'train', max_samples: int = None):
        import h5py
        split_map = {'train': 0, 'val': 1, 'test': 2}

        with h5py.File(hdf5_path, 'r') as f:
            split_col = f['split'][:]
            mask = split_col == split_map[split]
            self.spectra = f['spectrum'][:][mask]           # (N, 2, 60)
            self.smiles = [s.decode() if isinstance(s, bytes) else s
                           for s, m in zip(f['smiles'][:], mask) if m]

        # Limit samples (for fast testing)
        if max_samples and len(self) > max_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(self), max_samples, replace=False)
            idx.sort()
            self.spectra = self.spectra[idx]
            self.smiles = [self.smiles[i] for i in idx]

        print(f'[Dataset] {split}: {len(self)} spectra')

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        # Spectrum: (2, 60) → (60, 2) where last dim = [mz, intensity]
        spec = torch.from_numpy(self.spectra[idx]).float().permute(1, 0)  # (60, 2)

        # Generate random mask
        n_mask = max(1, int(N_PEAKS * MASK_FRAC))
        mask = torch.zeros(N_PEAKS, dtype=torch.bool)
        perm = torch.randperm(N_PEAKS)
        mask[perm[:n_mask]] = True

        # Create masked version (DreaMS convention: masked peaks → MASK_VAL)
        spec_mask = spec.clone()
        spec_mask[mask] = MASK_VAL

        # MACCS fingerprint
        maccs = smi_to_maccs(self.smiles[idx])

        return {
            'spec_mask': spec_mask,   # (60, 2)
            'spec_real': spec,        # (60, 2)
            'mask': mask,             # (60,)
            'maccs': maccs,           # (167,)
        }


def collate_fn(batch):
    """Collate list of dicts → batched dict."""
    return {
        'spec_mask': torch.stack([b['spec_mask'] for b in batch]),   # (B, 60, 2)
        'spec_real': torch.stack([b['spec_real'] for b in batch]),   # (B, 60, 2)
        'mask':      torch.stack([b['mask'] for b in batch]),        # (B, 60)
        'maccs':     torch.stack([b['maccs'] for b in batch]),       # (B, 167)
    }


def load_dreams(checkpoint_path: str, device: torch.device) -> DreaMS:
    """Load pre-trained DreaMS from checkpoint."""
    print(f'Loading DreaMS from {checkpoint_path} ...')
    raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = raw['state_dict']
    hp = raw['hyper_parameters']['args']
    if hasattr(hp, '__dict__'):
        hp = vars(hp)

    dformat = DataFormatA()
    spec_preproc = SpectrumPreprocessor(dformat=dformat, n_highest_peaks=N_PEAKS)

    clean = {'dformat': dformat, 'no_transformer_bias': True}
    for k, v in hp.items():
        clean[k] = str(v) if isinstance(v, type(Path())) else v
    clean['enable_cond_tokens'] = False

    model = DreaMS(Namespace(**clean), spec_preproc)
    load_sd = {k: v for k, v in sd.items()
               if not any(x in k for x in ['ro_out', 'mz_masking'])}
    model.load_state_dict(load_sd, strict=False)
    model = model.to(device)
    model.eval()
    print(f'  Params: {sum(p.numel() for p in model.parameters()):,}')
    return model


def train():
    # ── Args (simple, no argparse for now) ──
    HDF5_PATH = '/root/datasets/pairs_ready.hdf5'
    CKPT_PATH = '/root/DreaMS/dreams/models/pretrained/ssl_model.ckpt'
    timestamp = time.strftime('%m%d_%H%M')
    OUTPUT_DIR = f'/root/DreaMS/outputs/dreams_finetune_{timestamp}'
    BATCH_SIZE = 64
    MAX_EPOCHS = 10
    LR = 1e-4
    WEIGHT_DECAY = 0.01
    VAL_EVERY = 1        # validate every N epochs
    MAX_TRAIN_SAMPLES = None   # use full dataset
    MAX_VAL_SAMPLES = 2000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Log file (Tee: stdout → terminal + log file) ──
    log_path = os.path.join(OUTPUT_DIR, 'training.log')
    log_file = open(log_path, 'w', buffering=1)
    original_stdout = sys.stdout

    class Tee:
        def write(self, text): original_stdout.write(text); log_file.write(text)
        def flush(self): original_stdout.flush(); log_file.flush()
    sys.stdout = Tee()
    print(f'Output dir: {OUTPUT_DIR}')
    print(f'Log: {log_path}')

    # ── Datasets ──
    train_ds = SpectraMaccsDataset(HDF5_PATH, split='train', max_samples=MAX_TRAIN_SAMPLES)
    val_ds = SpectraMaccsDataset(HDF5_PATH, split='val', max_samples=MAX_VAL_SAMPLES)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    print(f'  Train: {len(train_ds)} samples ({len(train_loader)} batches)')
    print(f'  Val:   {len(val_ds)} samples ({len(val_loader)} batches)')

    # ── Model ──
    encoder = load_dreams(CKPT_PATH, device)
    model = DreaMS_ChemFinetuner(encoder).to(device)

    # DreaMS backbone is unfrozen — gradients flow back to update encoder weights,
    # so the (B, 60, 1024) output embeddings learn chemical semantics from MACCS.

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'Trainable: {trainable:,} / {total:,} params ({100*trainable/total:.1f}%)')
    print(f'  MACCS head: {sum(p.numel() for p in model.maccs_head.parameters()):,} params')

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY,
    )

    # ── Training loop ──
    best_val_loss = float('inf')
    timestamp = time.strftime('%m%d_%H%M')

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_losses = {'total': 0, 'masked_peak': 0, 'maccs': 0}
        n_batches = 0
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{MAX_EPOCHS}')
        for batch in pbar:
            spec_mask = batch['spec_mask'].to(device)
            spec_real = batch['spec_real'].to(device)
            mask = batch['mask'].to(device)
            maccs = batch['maccs'].to(device)

            optimizer.zero_grad()
            output = model(
                spec_mask=spec_mask,
                spec_real=spec_real,
                mask=mask,
                maccs_labels=maccs,
            )
            output['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()

            epoch_losses['total'] += output['total_loss'].item()
            epoch_losses['masked_peak'] += output['masked_peak_loss'].item()
            epoch_losses['maccs'] += output['maccs_loss'].item()
            n_batches += 1

            pbar.set_postfix({
                'loss': f"{output['total_loss'].item():.4f}",
                'maccs': f"{output['maccs_loss'].item():.4f}",
                'ssl': f"{output['masked_peak_loss'].item():.4f}",
            })

        # Epoch summary
        avg = {k: v / n_batches for k, v in epoch_losses.items()}
        elapsed = time.time() - t0
        print(f'\n  Epoch {epoch} | total={avg["total"]:.4f} '
              f'ssl={avg["masked_peak"]:.4f} maccs={avg["maccs"]:.4f} | {elapsed:.0f}s')

        # ── Validation ──
        if epoch % VAL_EVERY == 0:
            model.eval()
            val_losses = {'total': 0, 'maccs': 0}
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    spec_mask = batch['spec_mask'].to(device)
                    spec_real = batch['spec_real'].to(device)
                    mask = batch['mask'].to(device)
                    maccs = batch['maccs'].to(device)

                    output = model(
                        spec_mask=spec_mask,
                        spec_real=spec_real,
                        mask=mask,
                        maccs_labels=maccs,
                    )
                    val_losses['total'] += output['total_loss'].item()
                    val_losses['maccs'] += output['maccs_loss'].item()
                    val_n += 1

            val_avg = {k: v / val_n for k, v in val_losses.items()}
            print(f'  Val      | total={val_avg["total"]:.4f} maccs={val_avg["maccs"]:.4f}')

            # Save checkpoint
            is_best = val_avg['total'] < best_val_loss
            if is_best:
                best_val_loss = val_avg['total']

            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg['total'],
                'val_loss': val_avg['total'],
                'config': {
                    'lr': LR,
                    'batch_size': BATCH_SIZE,
                    'max_epochs': MAX_EPOCHS,
                    'n_train': len(train_ds),
                    'n_val': len(val_ds),
                },
            }
            ckpt_path = os.path.join(OUTPUT_DIR, f'finetuner_epoch{epoch}_{timestamp}.pt')
            torch.save(ckpt, ckpt_path)
            if is_best:
                best_path = os.path.join(OUTPUT_DIR, f'finetuner_best_{timestamp}.pt')
                torch.save(ckpt, best_path)
                print(f'  ✓ New best! Saved to {best_path}')

    print(f'\n=== Done! Best val_loss: {best_val_loss:.4f} ===')
    print(f'Outputs: {OUTPUT_DIR}')


if __name__ == '__main__':
    train()
