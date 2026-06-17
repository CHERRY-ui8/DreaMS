# Cond-Token DreaMS: Training Commands & Experiment Guide

记录了所有用到的训练命令和实验设计说明，方便快速复现。


## 训练命令

### 1. 带 cond tokens + mask_mz

```bash
python -m dreams.training.train \
      --run_name cond_100ep --project_name dreams_cond --job_key v1 \
      --dataset_pth /root/datasets/dreams_ready.hdf5 --dformat A \
      --batch_size 512 --train_objective mask_mz --train_regime pre-training \
      --lr 1e-4 --max_epochs 100 --no_val --mask_peaks \
      --enable_cond_tokens --pre_norm --no_transformer_bias --graphormer_mz_diffs \
      --wandb_entity_name 2693275288-peking-university
```

### 2. 不带 cond tokens + mask_mz

```bash
python -m dreams.training.train \
      --run_name baseline_100ep --project_name dreams_cond --job_key baseline \
      --dataset_pth /root/datasets/dreams_ready.hdf5 --dformat A \
      --batch_size 512 --train_objective mask_mz --train_regime pre-training \
      --lr 1e-4 --max_epochs 100 --no_val --mask_peaks \
      --pre_norm --no_transformer_bias --graphormer_mz_diffs \
      --wandb_entity_name 2693275288-peking-university
```

### 3. 带 cond tokens + mask_mz_hot

```bash
python -m dreams.training.train \
      --run_name cond_100ep --project_name dreams_cond --job_key v1 \
      --dataset_pth /root/datasets/dreams_ready.hdf5 --dformat A \
      --batch_size 512 --train_objective mask_mz_hot --train_regime pre-training \
      --lr 1e-4 --max_epochs 100 --no_val --mask_peaks \
      --enable_cond_tokens --pre_norm --no_transformer_bias --graphormer_mz_diffs \
      --wandb_entity_name 2693275288-peking-university
```

### 4. 不带 cond tokens + mask_mz_hot

```bash
python -m dreams.training.train \
      --run_name baseline_100ep --project_name dreams_cond --job_key baseline \
      --dataset_pth /root/datasets/dreams_ready.hdf5 --dformat A \
      --batch_size 512 --train_objective mask_mz_hot --train_regime pre-training \
      --lr 1e-4 --max_epochs 100 --no_val --mask_peaks \
      --pre_norm --no_transformer_bias --graphormer_mz_diffs \
      --wandb_entity_name 2693275288-peking-university
```

### 已运行的实验（v1日志）

两个实验都用 `--job_key v1`，共享目录 `dreams_cond/v1/`：

| CSV Version | 时间 | mask_mz_hot? | cond tokens? | Focal γ | 跑了多少epoch |
|-------------|------|-------------|-------------|---------|--------------|
| version_0   | 06:14 | ❌ (mask_mz) | ✅ | 0 | ~6 (测试) |
| version_2   | 07:57 | ✅ | ✅ | 5.0 | **44 epoch** |
| version_3   | 15:31 | ✅ | ❌ | 5.0 | **100 epoch** |

---

## Experiment A: Cond tokens vs baseline

- Train baseline (`enable_cond_tokens=False`) on dreams_ready.hdf5
- Train with cond tokens (`enable_cond_tokens=True`)
- Compare: loss curves, final validation loss, rare adduct performance

已做的分析：`/root/DreaMS/csv_analysis_results.txt`

---

## Experiment B: CE/Adduct ablation

四个变体，各跑10个epoch：
1. **both**: `--enable_cond_tokens`（adduct + CE都保留）
2. **neither**: 不加 `--enable_cond_tokens`
3. **adduct only**: 加 `--enable_cond_tokens`，但在forward中把CE token置零
4. **CE only**: 加 `--enable_cond_tokens`，但在forward中把adduct token置零

Metric: masked m/z prediction accuracy on test split

---

## Experiment C: Attention interpretability

对于已知谱（例如同一脂质 `[M+H]+` 在CE=35 vs CE=60）：
- 提取碎片peak到adduct token和CE token之间的attention权重
- 分析：高CE谱是否对CE token的attention不同于低CE谱？

工具：`/root/DreaMS/analyze_attention.py`

```bash
# 快速运行
python /root/DreaMS/analyze_attention.py --device cuda --n_samples 20

# 指定谱索引
python /root/DreaMS/analyze_attention.py --indices 5,7,11,190
```

---

## Experiment D: Zero-shot transfer to NIST20

- 用cond token模型（在GeMS上训练）
- 在NIST20谱上做推理（预测adduct + CE）
- 对比cond token模型与原版模型的embedding质量（retrieval accuracy）
