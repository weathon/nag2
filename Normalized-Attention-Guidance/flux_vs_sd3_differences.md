# Flux vs SD3 NAG 实现对比

## 一、Pipeline 层面差异

### 1. 基础架构

| 方面 | Flux | SD3 |
|---|---|---|
| 基类 | `FluxPipeline` | `StableDiffusion3Pipeline` |
| 文本编码器数量 | 2个（`prompt`, `prompt_2`） | 3个（`prompt`, `prompt_2`, `prompt_3`），额外支持 `clip_skip` |
| 最大序列长度默认值 | 512 | 256 |
| 默认 `guidance_scale` | 3.5 | 7.0 |

### 2. CFG / 负向提示词处理

- **Flux**：使用 `true_cfg_scale`，需要**额外的第二次前向传播**来处理负向提示词，默认关闭（值为 `1.0`）。因为 Flux 已将 guidance 蒸馏进模型的 guidance embeds，所以标准 CFG 需要显式的额外前向传播。
- **SD3**：使用标准批量 CFG —— 将 `[uncond, cond]` 的 latents 拼接后做一次联合前向传播，再拆分并计算 `uncond + scale * (cond - uncond)`。此外支持 `skip_guidance_layers`（SD3.5 Medium 的特性），Flux 中完全没有对应实现。

### 3. NAG 默认参数

| 参数 | Flux | SD3 |
|---|---|---|
| `nag_alpha` | 0.25 | 0.125 |
| `nag_guidance_type` | `"cfg_norm"` | `"angular"` |
| `nag_tau` | 2.5 | 2.5 |

### 4. `nag_end` 时序控制（仅 Flux）

Flux 有 `nag_end: float` 参数。在去噪循环中，当 `t < (1 - nag_end) * 1000` 时，会恢复原始的 attn processors，从而让 NAG 只作用于前段时间步。SD3 没有此机制，NAG 贯穿所有时间步，attn processors 在循环结束后才恢复。

### 5. Pooled Embeddings 的拼接方式

- **Flux**：简单拼接 `[positive_pooled, nag_negative_pooled]`
- **SD3**：更复杂 —— 需要兼顾 CFG batch 本身已经是 2× 或 1× 的情况。CFG 启用时追加 `original_pooled_prompt_embeds`（仅正向），否则直接复制当前 pooled embeds

### 6. Transformer 输入

- **Flux**：额外传入 `txt_ids`、`img_ids`、`guidance` embeds，且 `timestep` 除以 1000 后传入
- **SD3**：无 `txt_ids`/`img_ids`/`guidance`，输入更简洁，`timestep` 直接传入（不缩放）

### 7. VAE 解码

- **Flux**：先调用 `_unpack_latents()`（将 patch 还原），再做 `/ scaling_factor + shift_factor`
- **SD3**：直接 `/ scaling_factor + shift_factor`，无 unpack 步骤

### 8. 被 patch 的归一化层类型

- **Flux**：替换 2 种 —— `AdaLayerNormZero`、`AdaLayerNormContinuous`
- **SD3**：替换 4 种 —— 额外包含 `AdaLayerNorm` 和 `SD35AdaLayerNormZeroX`（SD3.5 特有的归一化变体）

---

## 二、Attention Processor 层面差异

### 1. 单流块支持（仅 Flux）

`NAGFluxAttnProcessor2_0` 同时处理 Flux 的**双流块**（有 `encoder_hidden_states`）和**单流块**（无 `encoder_hidden_states`）。在单流块中，guidance 只作用于**图像 token 切片**（通过 `encoder_hidden_states_length` 定位，跳过文本 token 位置）。SD3 的 `NAGJointAttnProcessor2_0` 只处理联合（双流）注意力。

### 2. Batch 倍数处理

- **Flux**：严格 2× batch（正向 + 负向）。在 reshape 前对 3D tensor 做 tile：`query.tile(2, 1, 1)`
- **SD3**：兼容 2×、3×、4× batch，以适配 CFG × NAG 的不同组合。在 reshape+transpose 后对 4D tensor 做 tile：`query.tile(2, 1, 1, 1)`

### 3. `cfg_norm` 中使用的范数类型

- **Flux**：L2 范数（`p=2`）
- **SD3**：L1 范数（`p=1`）

### 4. `angular_guidance` 公式

两个文件中的 `angular_guidance` 函数实现方式不同：

- **Flux**（[attention_flux_nag.py:15](nag/attention_flux_nag.py#L15)）：使用 `softclip` 对旋转角度进行软截断（`phi = softclip(scale * theta, tau * theta)`），最终结果乘以 `norm_c * 1.0`
- **SD3**（[attention_joint_nag.py:8](nag/attention_joint_nag.py#L8)）：更简单的线性公式 —— `sin((1 + alpha * s1) * theta)` 和 `sin(alpha * s2 * theta)`，最终结果乘以 `norm_c * 2.0`

> **注意**：两个实现中，`"angular"` 引导类型在 `__call__` 里目前均为**直通/拼接占位**（核心数学代码被注释掉）；只有 `"cfg_norm"` 处于实际运行状态。

### 5. `_set_nag_attn_processor` 接口差异

- **Flux**：需要额外传入 `encoder_hidden_states_length`（单流块中用于定位图像 token 的起始位置）
- **SD3**：无此参数

---

## 三、正确性分析：以 SD3 为基准，Flux 是否正确？

> 忽略架构特有细节（文本编码器数量、VAE 解码、Transformer 输入格式等），专注于 **NAG 引导逻辑本身**的等价性与 bug。

### 结论：`cfg_norm` 路径逻辑等价，但存在数值稳定性 bug

#### 共同逻辑（两者等价）

两者的 NAG batch 结构完全相同：
- **图像 latents** (`hidden_states`)：始终 batch=1（仅正向）
- **文本 conditioning** (`encoder_hidden_states`)：batch=2（正向 + NAG 负向）
- 每一层在 attention 内部将图像 token tile 为 2×，与两份文本条件分别做 attention，得到正向和负向的注意力输出，再对图像部分施加 `cfg_norm` 引导

`cfg_norm` 核心公式在两者中完全一致：
```
guidance = positive * scale - negative * (scale - 1)
ratio    = norm(guidance) / norm(positive)          # 范数比
output   = guidance * min(ratio, tau) / ratio        # 范数截断
output   = output * alpha + positive * (1 - alpha)  # alpha 混合
```

#### Bug 1：Flux 缺少 epsilon 保护（数值不稳定）

SD3 在两处分母加了 `1e-7`，Flux 两处均未加：

| 位置 | SD3 | Flux |
|---|---|---|
| 范数比分母 | `norm(positive) + 1e-7` | `norm(positive)`（无保护） |
| 最终除以 ratio | `/ (ratio + 1e-7)` | `/ ratio`（无保护） |

当 `norm(positive) ≈ 0` 或 `ratio ≈ 0` 时，Flux 会产生 `NaN`/`Inf`，SD3 不会。

影响位置：
- 双流块：[attention_flux_nag.py:170-171](nag/attention_flux_nag.py#L170-L171)
- 单流块：[attention_flux_nag.py:200-201](nag/attention_flux_nag.py#L200-L201)

#### Bug 2：双流块 `angular` 路径的 alpha 混合维度错误（已禁用，潜在 bug）

当 `guidance_type == "angular"` 时（[attention_flux_nag.py:164-165](nag/attention_flux_nag.py#L164-L165)）：
```python
hidden_states_guidance = torch.cat([hidden_states_positive, hidden_states_negative], dim=0)
# → shape [2, img_seq, dim]
hidden_states = hidden_states_guidance * alpha + hidden_states_positive * (1 - alpha)
# → hidden_states_positive 是 [1, img_seq, dim]，广播后结果 shape 为 [2, ...]，错误
```
SD3 的 angular 路径（也是占位）则直接跳过 alpha 混合，不会产生此问题。此路径当前默认未启用，但一旦启用会输出 shape 翻倍的错误结果。

#### 单流块的图像 token 覆写（正确，非 bug）

[attention_flux_nag.py:205-206](nag/attention_flux_nag.py#L205-L206) 将引导后的图像 token 同时写入正向和负向 batch：

```python
hidden_states_negative[:, encoder_hidden_states_length:] = image_hidden_states
hidden_states[:, encoder_hidden_states_length:]          = image_hidden_states
```

这与 SD3 的行为等价：SD3 也只把引导后的单份图像 token 传给下一层，下一层 tile 后，正负两份图像 token 起点相同（均为上一层的引导输出）。Flux 单流块同样如此，逻辑正确。

---

## 四、总结

| 维度 | Flux | SD3 |
|---|---|---|
| CFG 实现方式 | 显式第二次前向传播 | 批量拼接一次前向传播 |
| NAG 时序控制 | 支持 `nag_end` 中途关闭 | 全程开启 |
| 默认引导类型 | `cfg_norm` | `angular` |
| 注意力流类型 | 双流 + 单流均处理 | 仅双流 |
| 范数类型 | L2 | L1 |
| 归一化层 patch | 2 种 | 4 种 |
| Angular 公式 | softclip 软截断 | 线性简化版 |
| 文本编码器 | 2 个 | 3 个 |
