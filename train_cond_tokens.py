"""
train_cond_tokens.py — 训练 cond token 层 (adduct_embedding + ce_mlp)
加载 ssl_model.ckpt backbone (冻结), 只训练 1.2M cond params

用法:
  python3 train_cond_tokens.py --epochs 5 --lr 1e-4

输出:
  - checkpoints/cond_token_adapted.ckpt
  - csv_logs/train_log.csv
  - 控制台打印训练loss + 跨CE不变性评估
"""
import sys, types, os, argparse, warnings, time
from pathlib import Path
warnings.filterwarnings('ignore')

import torch, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from argparse import Namespace

# PyTorch MUST init CUDA before TensorFlow (imported via dreams) grabs the context
if torch.cuda.is_available():
    _ = torch.zeros(1, device='cuda')
    del _

# ── 0. msml mock ──
import dreams.utils.data as du; import dreams.utils.dformats as dformats
import dreams.utils.spectra as su; import dreams.models.dreams.dreams as dm
import dreams.models.dreams.layers as dl
import dreams.models.layers.fourier_features as ff
import dreams.models.layers.feed_forward as fw
for ns in ['msml','msml.models','msml.models.dreams','msml.models.layers','msml.utils']:
    sys.modules[ns] = types.ModuleType(ns)
sys.modules['msml.models.dreams.dreams'] = dm
sys.modules['msml.models.dreams.layers'] = dl
sys.modules['msml.models.layers.fourier_features'] = ff
sys.modules['msml.models.layers.feed_forward'] = fw
sys.modules['msml.utils.data'] = du; sys.modules['msml.utils.dformats'] = dformats
sys.modules['msml.utils.spectra'] = su

from dreams.models.dreams.dreams import DreaMS
from dreams.utils.data import SpectrumPreprocessor, MaskedSpectraDataset, MSData
from dreams.utils.dformats import DataFormatA
from dreams.definitions import SPECTRUM, ADDUCT, COLLISION_ENERGY, PRECURSOR_MZ

torch.set_grad_enabled(True)

# ── Args ──
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--max_batches', type=int, default=0, help='0 = full dataset')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
print(f'Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}')

# ── 1. Load ssl_model ──
CKPT = 'dreams/models/pretrained/ssl_model.ckpt'
raw = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = raw['state_dict']
hp = raw['hyper_parameters']['args']
if hasattr(hp, '__dict__'): hp = vars(hp)

# Build args
dformat = DataFormatA()
spec_preproc = SpectrumPreprocessor(dformat=dformat, n_highest_peaks=60)

clean = {'dformat': dformat, 'no_transformer_bias': True}
for k, v in hp.items():
    if isinstance(v, Path): clean[k] = str(v)
    else: clean[k] = v

# ── 2. Build model: backbone + ff_out + cond tokens ──
clean['enable_cond_tokens'] = True
clean['adduct_vocab_size'] = 101
clean['ce_max'] = 200.0

model = DreaMS(Namespace(**clean), spec_preproc)

# Load backbone + ff_out (everything except ro_out + mz_masking_loss)
load_keys = [k for k in sd if not any(x in k for x in ['ro_out', 'mz_masking'])]
load_sd = {k: sd[k] for k in load_keys}
missing, unexpected = model.load_state_dict(load_sd, strict=False)
print(f'\nModel loaded: {sum(p.numel() for p in model.parameters()):,} total params')
print(f'  Missing (expected = cond layers):')
for k in sorted(missing):
    print(f'    {k}')
assert all('adduct' in k or 'ce_mlp' in k or 'ro_out' in k for k in missing), \
    f'Unexpected missing keys: {[k for k in missing if "adduct" not in k and "ce_mlp" not in k and "ro_out" not in k]}'

# ── 3. Freeze backbone ──
for name, p in model.named_parameters():
    if 'adduct_embedding' in name or 'ce_mlp' in name:
        p.requires_grad = True
    else:
        p.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'\nTrainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)')
print(f'  adduct_embedding.weight: requires_grad={model.adduct_embedding.weight.requires_grad}')
print(f'  ce_mlp params: {sum(p.numel() for p in model.ce_mlp.parameters())}')

model = model.to(device)
model.train()

# ── 4. Dataset ──
HDF5 = Path('/root/datasets/dreams_ready.hdf5')
dataset = MaskedSpectraDataset(
    in_pth=HDF5, dformat=dformat, ssl_objective='mask_mz_hot',
    spec_preproc=spec_preproc, mask_peaks=True, frac_masks=0.3,
    min_n_masks=2, n_samples=None, mask_val=-1.,
    mask_intens_strategy='intens_cutoff', min_mask_intens=0.1,
    deterministic_mask=True,
    enable_cond_tokens=True, ce_max=200.0,
)

# Verify dataset works
sample = dataset[0]
print(f'\nDataset: {len(dataset)} spectra, vocab={len(dataset.adduct_vocab)} adducts')
assert ADDUCT in sample, f'Missing {ADDUCT} in dataset sample'
assert COLLISION_ENERGY in sample, f'Missing {COLLISION_ENERGY} in dataset sample'

# DataLoader with proper collation
def collate_fn(batch):
    out = {}
    for key in batch[0].keys():
        vals = [b[key] for b in batch]
        if key == 'spec_mask':
            out[key] = torch.from_numpy(np.stack(vals)).float()
        elif key == 'spec_real':
            out[key] = torch.from_numpy(np.stack(vals)).float()
        elif key == 'mask':
            out[key] = torch.from_numpy(np.stack(vals)).bool()
        elif key == ADDUCT:
            out[key] = torch.from_numpy(np.stack(vals)).long()
        elif key == COLLISION_ENERGY:
            out[key] = torch.from_numpy(np.stack(vals)).float()
        else:
            out[key] = torch.tensor(vals)
    return out

loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=0, drop_last=False, collate_fn=collate_fn)

# ── 5. Optimizer ──
cond_params = [p for n, p in model.named_parameters() if p.requires_grad]
optimizer = torch.optim.Adam(cond_params, lr=args.lr, weight_decay=0.0)

# ── 6. Training loop ──
os.makedirs('csv_logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
log_file = 'csv_logs/cond_token_train.csv'

print(f'\n{"="*60}')
print(f'Training cond tokens: {args.epochs} epochs, {trainable:,} params')
print(f'{"="*60}')

with open(log_file, 'w') as f:
    f.write('epoch,batch,loss,time\n')

total_batches = len(loader)
if args.max_batches > 0:
    total_batches = min(total_batches, args.max_batches)

for epoch in range(args.epochs):
    epoch_loss = 0.0
    epoch_start = time.time()
    
    for i, batch in enumerate(loader):
        if args.max_batches > 0 and i >= args.max_batches:
            break
        
        spec_mask = batch['spec_mask'].to(device)
        spec_real = batch['spec_real'].to(device)
        mask = batch['mask'].to(device)
        adduct = batch[ADDUCT].to(device)
        ce = batch[COLLISION_ENERGY].to(device)
        
        optimizer.zero_grad()
        loss, embs, pred_mz, real_mz = model.spec_ssl_step(
            spec_mask, spec_real, mask, charge=None,
            adduct=adduct, collision_energy=ce
        )
        loss = loss.sum() / loss.numel()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if i % 200 == 0:
            elapsed = time.time() - epoch_start
            print(f'  E{epoch} B{i}/{total_batches} loss={loss.item():.4f} '
                  f'({i/total_batches*100:.0f}%, {elapsed:.0f}s)')
            with open(log_file, 'a') as f:
                f.write(f'{epoch},{i},{loss.item():.6f},{elapsed:.0f}\n')
    
    avg_loss = epoch_loss / total_batches
    epoch_time = time.time() - epoch_start
    print(f'>>> Epoch {epoch} done: avg_loss={avg_loss:.4f}, time={epoch_time:.0f}s')
    with open(log_file, 'a') as f:
        f.write(f'{epoch},-1,{avg_loss:.6f},{epoch_time:.0f}\n')

# ── 7. Save checkpoint ──
save_path = 'checkpoints/cond_token_adapted.ckpt'
torch.save({
    'model_state_dict': model.state_dict(),
    'adduct_vocab': dataset.adduct_vocab,
    'adduct_vocab_size': len(dataset.adduct_vocab),
    'ce_max': 200.0,
    'args': clean,
}, save_path)
print(f'\nCheckpoint saved: {save_path} ({os.path.getsize(save_path)/1e9:.2f}GB)')

# ── 8. Evaluate: cross-CE invariance ──
print(f'\n{"="*60}')
print('Evaluating cross-CE invariance (precursor embedding)')
print(f'{"="*60}')

model.eval()
with torch.no_grad():
    msdata = MSData(HDF5, in_mem=True)
    ds_eval = msdata.to_torch_dataset(spec_preproc)
    
    import h5py
    with h5py.File(HDF5, 'r') as f:
        smiles_raw = f['smiles'][:]; ce_raw = f['collision_energy'][:]
        adducts_raw = f['adduct'][:]; prec_mz_raw = f['precursor_mz'][:]
    
    adduct_strs = np.array([a.decode() for a in adducts_raw])
    mh_idx = np.where(adduct_strs == '[M+H]+')[0]
    
    # Build lookup
    from collections import defaultdict
    smiles_dict = defaultdict(list)
    for i in mh_idx:
        sm = smiles_raw[i].decode()
        if len(sm) > 5:
            smiles_dict[sm].append((int(i), float(ce_raw[i])))
    
    # Pick 3 test molecules
    test_mols = []
    for sm, entries in sorted(smiles_dict.items(), key=lambda x: -len(x[1])):
        ces = sorted(set(c for _, c in entries))
        if len(ces) >= 4:
            picks = [ces[0], ces[len(ces)//3], ces[2*len(ces)//3], ces[-1]]
            idxs, actual_ces = [], []
            for t in picks:
                best = min(entries, key=lambda e: abs(e[1] - t))
                if best[0] not in idxs:
                    idxs.append(best[0]); actual_ces.append(best[1])
            if len(set(actual_ces)) >= 3:
                test_mols.append((sm, idxs, actual_ces))
        if len(test_mols) >= 3:
            break
    
    # Also add [M+Na]+ and [M-H]-
    for ad_name in ['[M+Na]+', '[M-H]-']:
        mask = adduct_strs == ad_name
        idxs = np.where(mask)[0]; ces = ce_raw[idxs]
        if len(idxs) >= 4:
            sorted_i = idxs[np.argsort(ces)]
            picks_idx = [int(sorted_i[i]) for i in [0, len(sorted_i)//3, 2*len(sorted_i)//3, -1]]
            test_mols.append((f'[{ad_name}]', picks_idx, [float(ce_raw[i]) for i in picks_idx]))
    
    vocab = ['<PAD>', '<UNK>'] + sorted(set(adduct_strs))
    str2idx = {s: i for i, s in enumerate(vocab)}
    
    from torch.utils.data import Subset
    print(f'{"Molecule":>30} | {"CEs":>16} | {"Sim(cond)":>9} | {"Sim(no_init)":>12}')
    print('-' * 75)
    
    for mol_name, idxs, actual_ces in test_mols:
        subset = Subset(ds_eval, idxs)
        ldr = DataLoader(subset, batch_size=len(idxs), shuffle=False, drop_last=False)
        batch = next(iter(ldr))
        spec = batch[SPECTRUM].to(device)
        
        ad_name = mol_name if mol_name in ['[M+Na]+', '[M-H]-'] else '[M+H]+'
        ad_t = torch.tensor([str2idx.get(ad_name, 1)], device=device).long().expand(len(idxs))
        ce_t = torch.tensor(actual_ces, device=device).float()
        
        embs = model(spec, charge=None, adduct=ad_t, collision_energy=ce_t)
        prec_emb = embs[:, 0, :]
        
        n = len(idxs)
        sim = torch.nn.functional.cosine_similarity(
            prec_emb.unsqueeze(1), prec_emb.unsqueeze(0), dim=-1)
        mask_ut = torch.triu(torch.ones(n, n), diagonal=1).bool()
        avg_sim = sim[mask_ut.to(device)].mean().item()
        
        short = mol_name[:28] + '..' if len(mol_name) > 28 else mol_name
        ce_str = ', '.join(f'{c:.0f}' for c in actual_ces)
        print(f'{short:>30} | {ce_str:>16} | {avg_sim:>8.4f} |')

print(f'\nDone! Checkpoint at {save_path}')
print(f'Training log at {log_file}')
print(f'To load this model later:')
print(f'  ckpt = torch.load(\"{save_path}\", ...)')
print(f'  model.load_state_dict(ckpt[\"model_state_dict\"])')
