# 架构改造四阶段路线图

## 训练策略

两阶段训练：**Phase 1** 冻结 T5 backbone，只训练 Projector + 可选 MACCS 分类头（对齐 MS 特征空间）；**Phase 2** 注入 LoRA，冻结 backbone，**解冻 Projector（小学习率，默认 lr×0.1）+ LoRA 一起训练**（适配 T5 解码）。避免全量微调，防止灾难性遗忘。

> **废弃**: D1 K-heads (`--projector_type k_heads`) 不再使用。当前只使用 MLP projector（`--projector_type mlp`，默认）。

## 实验结果

### 阶段零：5k 过拟合基线 ✅

在 5000 个训练样本上验证代码正确性：

| Phase | 配置 | 结果 | 说明 |
|-------|------|------|------|
| 1 (projector only) | MLP d=8, K=128, t5-small | Epoch 580: **97.80% exact match** (plateau) | 投影器容量足够，代码无 bug |
| 2 (projector + LoRA) | frozen projector, LoRA r=64 | Epoch 100: **99.22% exact**, Tanimoto 0.998 | LoRA 有效，收敛较慢 |
| 2 (unfrozen proj + LoRA) | **解冻 Projector (lr×0.1)** + LoRA r=64 | **Epoch 100: 99.56% exact**, 100% valid, Tanimoto 0.998 | 对比 frozen: unfrozen 100ep 即更高（99.56% vs 99.22%） |

目标：训练集 Loss → 接近 0，SMILES 生成准确率接近 100%。

### 阶段一：引入子结构监督 + LoRA 微调 — 全量数据 ✅

**配置**: `phase1_k16_d2_full_scratch_maccs01_0526_1324` → `phase2_k16_d2_full_resume_maccs01_lora64_uproj_0527_0347`
- MLP Projector: d=8, K=16（而非 5k 实验的 K=128，因为全量数据更大、vocab 够用）
- MACCS: α=1.0, β=0.1
- Phase 2: LoRA r=64 + unfrozen projector (lr×0.1), 10 epochs
- 数据集: ~314K 质谱-分子对，分子级无泄漏划分

#### Phase 1（对齐期，5 epochs）

| Epoch | Train Loss | Val Loss | CE / BCE | MACCS (train/val) |
|-------|-----------|---------|---------|------------------|
| 5 | 1.5899 | **1.3043** | 1.5573 / 0.3259 | 84.5% / **84.4%** |

**Generation eval (epoch 5)**: Valid SMILES 40.0% | Exact match **0.00%** | Tanimoto mean **0.0971**

#### Phase 2（LoRA 适配期，10 epochs）

| Epoch | Train Loss | Val Loss | CE / BCE | MACCS (train/val) |
|-------|-----------|---------|---------|------------------|
| 1 | 1.2017 | 0.9210 | 1.1688 / 0.3287 | 84.4% / 84.2% |
| 2 | 1.0136 | 0.8583 | 0.9810 / 0.3265 | 84.5% / 84.2% |
| 3 | 0.9547 | 0.8216 | 0.9222 / 0.3247 | 84.6% / 84.3% |
| 4 | 0.9198 | 0.8007 | 0.8875 / 0.3233 | 84.7% / 84.3% |
| 5 | 0.8968 | 0.7865 | 0.8646 / 0.3221 | 84.7% / **84.4%** |
| 6 | 0.8816 | 0.7762 | 0.8495 / 0.3211 | 84.8% / 84.4% |
| 7 | 0.8707 | 0.7700 | 0.8387 / 0.3203 | 84.8% / 84.4% |
| 8 | 0.8647 | 0.7673 | 0.8327 / 0.3197 | 84.9% / 84.4% |
| 9 | 0.8616 | 0.7655 | 0.8296 / 0.3193 | 84.9% / 84.5% |
| 10 | **0.8602** | **0.7651** | **0.8283 / 0.3191** | 84.9% / **84.5%** |

**Generation eval (epoch 5)**: Valid SMILES **84.0%** | Exact match **0.00%** | Tanimoto mean **0.1218**
**Generation eval (epoch 10)**: Valid SMILES **81.0%** | Exact match **0.00%** | Tanimoto mean **0.1113**

#### 关键观察

| 指标 | Phase 1 → Phase 2 变化 |
|------|----------------------|
| Val Loss | 1.30 → **0.77**（↓41%） |
| CE Loss | 1.56 → **0.83**（↓47%） |
| BCE Loss | 0.326 → **0.319**（↓2%，接近饱和） |
| MACCS val acc | 84.4% → **84.5%**（接近饱和） |
| Valid SMILES rate | 40.0% → **84.0%**（↑2.1×，Phase 2 epoch 5） |
| Tanimoto | 0.097 → **0.122**（↑26%，epoch 5） |

**结论**:
1. Phase 1 让 Projector 学到了有化学意义的特征（MACCS 84.4%），但 T5 生成还很差（40% valid, 0% exact）。
2. Phase 2 LoRA 适配后 val_loss 大幅下降（1.30→0.77），valid SMILES rate翻倍（40%→84%），但 exact match 仍为 0%。
3. MACCS 准确率在 Phase 2 中基本饱和（84.4%→84.5%），BCE loss 几乎不变——说明 BCE 梯度已耗尽，需要更强的主干或更多 epoch 才能进一步提升。
4. 生成方面：Tanimoto 在 epoch 5 达到 0.122 后，epoch 10 反而下降到 0.111——生成质量没有随 loss 下降而改善，暗示解码策略（greedy/beam search）或模型容量有瓶颈。

### 阶段二：架构升级，从 MLP 到 Q-Former (Upgrade to Q-Former)

目标： 提升特征提取的上限。MLP 只能做简单的线性/非线性映射，处理复杂质谱序列的能力有限。
- 操作：
  1. 将 MLP Projector 替换为 Q-Former（例如设置 32 个 Learnable Queries）。
  2. 让这 32 个 Queries 通过 Cross-Attention 去"查阅" DreaMS 输出的质谱序列特征。
  3. 分类头和 T5 现在都接在 Q-Former 的输出上。
- 核心逻辑： Q-Former 能够动态地关注质谱图中的特定峰群（比如 Query 1 专门盯着低质荷比的碎片，Query 2 盯着中性丢失）。
- 验收标准： 在 460k 数据上，对比阶段一，MACCS 分类准确率和 SMILES 生成的 BLEU/准确率应该有显著的跃升。

### 阶段二：架构升级 (Unlock MS Sequence + Q-Former) —— 重大修改！
原计划： 仅仅把 MLP 换成 Q-Former。
新计划： “释放序列特征” + Q-Former。
- 操作：
  - 修改 MS Encoder： 找到 MS Encoder 的代码，去掉最后的 Pooling 层（或 CLS token 提取），让它输出完整的质谱峰序列特征 (B, L, 1024)。
  - 接入 Q-Former： 让 Q-Former 的 32 个 Queries 去 Cross-Attend 这个 (B, L, 1024) 的序列。
为什么必须改： 这是从根本上解决 400k 数据泛化失败的钥匙。只有拿到序列，Q-Former 才能真正发挥“对齐”的作用，而不是在单向量上“瞎猜”。

### 阶段三：Prompt 引导生成 (Prompt-Conditioned Generation)

目标： 进一步降低 T5 的生成难度，限制"化学幻觉"。
- 操作：
  1. 在前向传播时，截获分类头预测概率高的 MACCS keys（例如 P>0.8P > 0.8P>0.8）。
  2. 将这些 keys 转化为特殊的 Token 或文本（例如 <MACCS_16> <MACCS_32>），拼接到 T5 的输入最前面。
  3. T5 现在的输入变成了：[预测的子结构 Prompt] + [Q-Former 提取的连续特征]。
- 核心逻辑： 相当于在让 T5 画图前，先给它一份"零件清单"。
- 验收标准： 评估生成的 SMILES。重点观察无效 SMILES（Invalid SMILES，即 RDKit 无法解析的字符串）的比例是否大幅下降。

### 阶段四：引入对比学习 (Contrastive Alignment)

目标： 逼迫模型学习更细粒度的结构差异（比如区分同分异构体）。
- 操作：
  1. 引入一个冻结的 1D 分子编码器（如 ChemBERTa 或 MoLFormer）来提取真实 SMILES 的特征。
  2. 计算 Q-Former 输出特征与分子编码器特征之间的 InfoNCE Loss。
  3. 此时总 Loss 变为：$$Loss = Loss_{InfoNCE} + Loss_{BCE} + Loss_{CE}$$
- 验收标准： 模型在测试集上的表现达到最优，特别是对于复杂代谢物的骨架预测更加准确。



## 各阶段与两阶段策略的结合指南
### 阶段一：引入子结构监督 (Add Substructure Loss)
- Phase 1 (对齐期): 冻结 T5。训练 MLP + 分类头。 
  - Loss = \alpha \cdot Loss_{CE} (通过冻结的 T5 传回梯度) + \beta \cdot Loss_{BCE} (MACCS 分类)。
  - 目的： 强迫 MLP 提取具有化学意义的特征。
- Phase 2 (适配期): 冻结 T5。训练 LoRA + **解冻 Projector（小学习率，lr×0.1）**。
  - Loss = Loss_{CE}。
  - 目的： 让 LoRA 和微调后的 Projector 一起配合，提升 SMILES 准确率。
### 阶段二：架构升级 (Upgrade to Q-Former)
- Phase 1 (对齐期): 冻结 T5。训练 Q-Former + 分类头。 
  - Loss = \alpha \cdot Loss_{CE} + \beta \cdot Loss_{BCE}。
  - 目的： Q-Former 参数较多，Phase 1 是它学习质谱特征交叉注意力（Cross-Attention）的关键时期。
- Phase 2 (适配期): 冻结 T5。训练 LoRA + **微调 Q-Former（小学习率）**。
### 阶段三：Prompt 引导生成 (Prompt-Conditioned Generation)
- 注意：这个阶段 Phase 2 是绝对的主力！
- Phase 1 (对齐期): 保持阶段二的训练方式，确保分类头能准确输出 MACCS keys。
- Phase 2 (适配期): 冻结 T5。训练 LoRA + **微调 Q-Former**。
  - 输入变化： T5 的输入现在变成了 [MACCS Prompt 文本] + [Q-Former 特征]。
  - 目的： LoRA 的核心任务变成了学习如何"听从" Prompt 的指令。 它需要学会根据文本提示中的子结构来约束 SMILES 的生成，避免化学幻觉。
### 阶段四：引入对比学习 (Contrastive Alignment)
- Phase 1 (对齐期): 冻结 T5，冻结 ChemBERTa。训练 Q-Former + 分类头。 
  - Loss = Loss_{InfoNCE} + Loss_{BCE} + Loss_{CE}。
  - 目的： 这是 Q-Former 特征提取能力的终极形态，对比学习将极大地优化其特征空间的排布。
- Phase 2 (适配期): 冻结 T5。训练 LoRA + **微调 Q-Former**。
  - 目的： 享受极致对齐后的特征，生成最精确的 SMILES。
