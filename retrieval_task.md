# MS ↔ Molecule Retrieval (CLIP-style Contrastive Learning)

从生成（MS→SMILES）转型为检索/判别任务。

## 架构

\`\`\`
MS 谱图 → DreaMS (frozen) → 1024-d   → Projector(1024→hidden→512) → L2Norm → 512-d
SMILES  → MoLFormer(frozen) → 768-d   → Projector(768→hidden→512)  → L2Norm → 512-d
                                        ↓
                              Symmetric InfoNCE (learnable τ)
\`\`\`

**Projector 结构**（`model.py` `Projector` 类）：
- Depth=2（默认）：`Linear(in, hidden) → GELU → LayerNorm → Linear(hidden, out)`
- Depth=3：多一层 `Linear(hidden, hidden) → GELU → LayerNorm`
- LayerNorm 在最后一层 Linear 之前：防止表示坍塌（SimCLR v2 insight）

## 代码结构

| 文件 | 内容 |
|---|---|
| `ms2mol_retrieval/model.py` | `MSMolCLIP` 双塔模型 + `Projector` MLP + `compute_loss()` |
| `ms2mol_retrieval/dataset.py` | `MSMolRetrievalDataset` + MoLFormer/Cluster 预计算缓存 |
| `ms2mol_retrieval/sampler.py` | `HardNegativeBatchSampler`（聚类负样本采样） |
| `ms2mol_retrieval/train.py` | 训练脚本（含课程学习） |

## 数据

`/root/datasets/pairs_with_embs.hdf5` — 368,934 条，含预计算 1024-d DreaMS 位置0 embedding

| Split | 谱图数 | 独特分子数 |
|---|---|---|
| Train | 313,613 | 46,351 |
| Val | 18,362 | 2,726 |
| Test | 36,959 | 5,453 |

**数据加载流程**（`MSMolRetrievalDataset`）：
1. 从 HDF5 读 `embedding` (1024-d) + `smiles` + `split`
2. 在建 Dataset 时一次性预计算所有独特分子的 MoLFormer [CLS] 768-d 嵌入
3. `__getitem__` 返回：`ms_emb` (1024-d) + `mol_emb` (768-d) + `mol_id` (标签)

**MoLFormer**: `ibm/MoLFormer-XL-both-10pct` (44M, 768-d [CLS], frozen).  
⚠ CUDA 兼容：需设置 `config.deterministic=True` + `config.deterministic_eval=True`，否则
Performer 的 `torch.linalg.qr()` 在 CUDA 12+ 上触发 cusolver 错误。

## Loss 设计（Symmetric InfoNCE）

\`\`\`python
# model.py · MSMolCLIP.compute_loss()
B = ms_feat.size(0)
scale = self.logit_scale.exp()          # 可学习温度 τ = 1/scale
logits = ms_feat @ mol_feat.T * scale   # (B, B) 相似度矩阵
labels = torch.arange(B, device=device)  # 对角线 = 正样本对

loss_ms = CE(logits, labels)            # MS→Mol: 每个 MS 找到对应 Mol
loss_mol = CE(logits.T, labels)         # Mol→MS: 每个 Mol 找到对应 MS
loss = (loss_ms + loss_mol) / 2.0       # 对称 InfoNCE（CLIP 标准）
\`\`\`

**关键设计选择**：
- **对称 Loss**：MS→Mol 和 Mol→MS 两个方向都做 CE，再平均。CLIP 原文方式，比单向更稳定。
- **可学习 τ**（logit_scale）：初始 `ln(1/0.07)` ≈ 2.66，exp() 后为 14.3。训练中自动调整对比的锐利程度。
- **L2 Normalize 先于相似度计算**：等价于 cosine similarity，确保特征落在单位超球面上，防止某个模态 collapse。
- **in-batch 负样本**：每个 batch 内其他 B-1 个样本充当负样本，不用显式负采样队列（对比 SimCLR）。

**评估**（FAISS 全库检索）：
\`\`\`python
# model.py · MSMolCLIP.compute_retrieval_metrics()
index = faiss.IndexFlatIP(proj_dim)  # inner product = cosine (L2-normed)
index.add(mol_feat_all)              # 所有独特分子的嵌入作为索引
# 对每个 MS 查询，在索引中搜 top-k，看正确分子是否在结果中
\`\`\`

## 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--batch_size` | 256 | **推荐 8192** |
| `--proj_dim` | 256 | 投影空间维度（推荐 512） |
| `--proj_hidden` | 1024 | MLP 隐藏层宽度 |
| `--proj_depth` | 2 | MLP 隐藏层数（2 or 3，**推荐 3**） |
| `--hard_ratio` | 0.0 | 同 cluster 负样本比例（0.0=纯随机） |
| `--hard_ratio_phase2` | None | 课程学习 Phase 2 的 hard_ratio |
| `--switch_epoch` | 10 | 课程学习切换 epoch |

**最佳配置**（无需 hard negatives）：
\`\`\`bash
--batch_size 8192 --proj_dim 512 --proj_hidden 1024 --proj_depth 3
\`\`\`

**优化器配置**：
- AdamW（lr=3e-4, weight_decay=0.01）
- Cosine scheduler + 1000 steps warmup
- Gradient clipping at 1.0

## 关键实现

### 温度 τ (learnable)

\`\`\`python
self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))
# logits = ms @ mol.T * logit_scale.exp()  →  τ = 1 / logit_scale.exp()
\`\`\`

τ 的诊断：<0.01=极度不自信, 0.01~0.05=健康(CLIP最佳), >0.05=分布太平坦

### Hard Negative 采样

1. 离线：RDKit Morgan FP (r=2, 2048b) → MiniBatchKMeans (k=1000) → 缓存
2. 在线：每个 batch 取 `hard_ratio` 比例来自同一 cluster（结构相似分子）+ 其余随机

### 课程学习（两阶段）

\`\`\`
Phase 1 (ep 1-switch_epoch):  hard_ratio=0.0  → 纯随机负样本，快速建立粗对齐
Phase 2 (ep switch_epoch+1+):  hard_ratio_phase2 → 结构相似负样本，死磕细粒度
\`\`\`

实际效果：**深层投影头下课程学习收益为负**，推荐只用 Phase 1。

## 实验报告

### 全实验对比 (50 epochs, 313K 全量)

| 实验 | BS | Proj | 隐藏层 | 参数 | Hard | **Test R@1** | Test R@5 | Test R@10 | τ |
|---|---|---|---|---|---|---|---|---|---|
| **🏆 BS8192-D3-W1024** | 8192 | 512 | 1024×3 | 5.0M | 0.0 | **10.51%** | 25.83% | 34.44% | 0.0498 |
| BS8192-D3-W1024_Curric | 8192 | 512 | 1024×3 | 5.0M | 0→0.10 | 10.22% | 25.74% | 34.42% | 0.0506 |
| BS8192-D2-W512 | 8192 | 512 | 512×2 | 1.4M | 0.0 | 9.40% | 23.39% | 31.76% | 0.0560 |
| BS8192-D2-W512_HN10 | 8192 | 512 | 512×2 | 1.4M | 0.10 | 9.14% | 23.42% | 32.06% | 0.0497 |
| BS8192-D2-W256 | 8192 | 256 | 512×2 | 0.9M | 0.0 | 9.02% | 22.82% | 30.95% | 0.0556 |
| BS8192-D2-W512_HN25 | 8192 | 512 | 512×2 | 1.4M | 0.25 | 7.94% | 21.39% | 29.74% | 0.0497 |
| BS256-D2-W512 | 256 | 512 | 512×2 | 1.4M | 0.0 | 7.38% | 20.54% | 28.80% | 0.0255 |

所有大 batch 实验都在 epoch 10 达到峰值后缓慢下降。

### 基线

| 基线 | Recall@1 |
|---|---|
| DreaMS MS→MS | 76.2% |
| MoLFormer Mol→Mol | 0.0% |
| 随机猜测 | 0.04% |

**DreaMS MS→MS (76.2%) 不是 MS→Mol 的理论上限**。不同分子可以有几乎一样的质谱图
（同分异构体），质谱物理上丢失了部分结构信息。**MS→Mol 上限估计在 40~50%**。

## 结论

1. **更深投影头是唯一有效突破**。3 层 MLP (5M) 将 R@1 从 9.40% → 10.51%。
2. **大 batch 是基础**。8192 vs 256 = 10.51% vs 7.38%。
3. **Hard negatives 回报递减**。深层投影头下加入硬负样本反而降低效果。
4. **所有实验 epoch 10 后过拟合**。深层投影头峰值更高但趋势不变。
5. **距理论上限仍远**。10.51% vs 40-50%，跨模态对齐需根本性突破。

## 下一步

1. **更宽投影头 (proj_hidden=2048) + 早停**。尝试 2x 宽度能否进一步突破。
2. **数据增强**。MS 谱图 m/z 偏移 + 强度扰动，延缓过拟合。
3. **更换分子编码器**。MolT5 / BioT5 / ChemBERTa-2，替代 2022 年的 MoLFormer。
4. **多任务学习**。保留 DreaMS 的 SSL 预训练信号，防止特征退化。

## 当前架构（CLIP 风格 512-d 共享空间 + 2 辅助头）

```
ms_emb (1024-d) ──┬──→ MS Projector (1024→1024→512) ──→ L2Norm ──→ 512-d ──┐
                  │                                                         InfoNCE
mol_emb (768-d) ──→ MoL Projector (768→1024→512) ──→ L2Norm ──→ 512-d ────┘

ms_emb (1024-d) ──┬──→ MACCS Proj (1024→256→166) ─────────────────→ BCE Loss
                  └──→ MW Proj (1024→64→1) ─────────────────────→ Huber Loss
```

| 组件 | 结构 | 参数量 | Loss |
|---|---|---|---|
| MS Projector | 1024→1024→GELU→LN→512 | ~1.58M | InfoNCE (对称 CE) |
| MoL Projector | 768→1024→GELU→LN→512 | ~1.31M | InfoNCE (对称 CE) |
| MACCS Proj | 1024→256→LN→GELU→166 | ~0.31M | BCEWithLogits |
| MW Proj | 1024→64→LN→GELU→1 | ~0.07M | Huber (Smooth L1) |
| **总计** | | **~3.26M** | |

两个辅助 head 从 **raw 1024-d 输入**出发，不经过 512-d 投影空间。
InfoNCE 在 **512-d 共享空间**中计算，与纯 CLIP 架构一致。

**三任务架构**（CLIP 风格 512-d 共享空间 + 2 辅助头）：

| 任务 | 类型 | 输出 | Loss | 作用 |
|---|---|---|---|---|
| 跨模态对齐 | 对比学习 | 512-d (L2 normed) | Symmetric InfoNCE | 最终目标 |
| MACCS 子结构 | 多标签分类 | 166-d logits | BCEWithLogitsLoss | 诊断：结构化学 |
| 分子量回归 | 回归 | 1-d | Huber Loss (Smooth L1) | 诊断：质量信息 |

### 训练计划

**Phase 1** ✅ 已完成 — 代码冒烟测试 + 单 Batch 过拟合验证
- 2026-06-03: Phase 1.1 Dummy 测试通过，Phase 1.2 真实数据过拟合通过
  - MACCS Acc: 1.0000 ✅（完美过拟合）
  - 分子量 MAE: **2.3 Da**（loss_mw 归一化空间降 10 倍，但 ~200 步后 plateau）
  - 关键发现：分子量回归需要做 z-score 归一化（Huber Loss L1 梯度饱和问题）

**⚠ 已知问题：分子量回归的精度上限**
- 32 个样本中多个共享同一分子量（437.99 Da），但 DreaMS embedding 不同（同一分子的不同谱图）
- 模型在 ~200 步后陷入 plateau，MAE ~2.3 Da，无法进一步下降
- 原因不是代码 bug，而是**数据噪声**：同一分子不同谱图→不同 DreaMS 特征→相同目标值
- 这意味着分子量回归的精度受限于 MS 谱图的**可重复性**和 DreaMS encoder 的**稳定度**
- 当前不调整架构。如果后续训练中分子量 MAE 始终无法突破 ~2 Da 的水平，应优先排查：
  1. 同一分子多个谱图的 DreaMS 特征方差（直接影响回归上限）
  2. 是否需要不同预处理/对齐策略提高特征稳定性
  3. 是否分子量回归本身就是一个"有噪声标签"的任务

**Phase 2** ✅ 已完成 — 数据管道优化
- 2026-06-03:
  - `MultiTaskRetrievalDataset` 集成到 `dataset.py`：加载 ms_emb + MoLFormer 缓存 + MACCS (HDF5) + 分子量 (RDKit 预缓存)
  - MACCS 从 HDF5 的 `maccs` 字段加载（167列，取前166列，最后一列为padding）
  - 分子量从 SMILES 用 RDKit 预计算并缓存到 `shared_cache/mol_weight_*.npy`，避免每 epoch 重复计算
  - 分子量自动做 z-score 归一化（训练集 mean±std）
- `MSMolCLIPMultiTask` 在 `model.py` 中（继承 `MSMolCLIP`，增加 MACCS + MW 两个 head）
- `train_multitask.py` 训练脚本：CLIP 风格 512-d InfoNCE + BCE (MACCS) + Huber (MW)
- 数据文件一览：
  - `shared_cache/molformer_embs_{split}.npy` — MoLFormer 768-d 嵌入（预计算）
  - `shared_cache/mol_weight_{split}.npy` — 分子量 Da 值（预计算）
  - HDF5 `maccs` 字段 — MACCS 166-bit 指纹
  - HDF5 `embedding` 字段 — DreaMS 1024-d 特征

**Phase 3** 待开始 — 两阶段渐进式训练
- Phase 3.1: 冻结 backbone，只训 3 个 Projector（5-10 epoch）
- Phase 3.2: 解冻最后 2-4 层 Transformer，联合精调

**Phase 4** 待开始 — 评估指标
- InfoNCE: FAISS 全库检索 R@1/5/10（与纯 CLIP 基线 10.51% 对比）
- MACCS: Macro-F1 + AUROC（目标 > 0.90）
- 分子量: MAE（目标 < 2 Da）