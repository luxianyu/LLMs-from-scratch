# 额外的分类微调实验（Additional Classification Finetuning Experiments）

下表展示了额外实验，用于回答关于模型设计选择的各类问题。第一行使用与本章主要实验相同的设置，作为参考。

- 例如：
  - 比较第 1 行与第 2 行可以回答问题：“训练最后一个输出 token 与第一个输出 token 的性能差异如何？”
  - 比较第 1 行与第 3 行可以回答问题：“只训练最后一层与训练最后一个 transformer block 的性能差异如何？”
  - 依此类推。

|      | 模型               | 权重       | 可训练 token 位置     | 可训练层        | 上下文长度                                           | 训练准确率 | 验证准确率 | 测试准确率 | 训练时间 | CPU/GPU |
| ---- | ----------------- | ---------- | -------------------- | --------------- | --------------------------------------------------- | ---------- | ---------- | ---------- | -------- | ------- |
| 1    | gpt2-small (124M)  | 预训练     | last                  | last_block      | 最长训练样本 (120)                                  | 96.63%     | 99.33%    | 95.00%     | 0.28 min | A100    |
| 2    | gpt2-small (124M)  | 预训练     | first                 | last_block      | 最长训练样本 (120)                                  | 78.46%     | 80.54%    | 75.00%     | 0.28 min | A100    |
| 3    | gpt2-small (124M)  | 预训练     | last                  | last_layer      | 最长训练样本 (120)                                  | 78.65%     | 79.87%    | 72.00%     | 0.25 min | A100    |
| 4    | gpt2-small (124M)  | 预训练     | last                  | last_two_blocks | 最长训练样本 (120)                                  | 98.85%     | 98.66%    | 98.33%     | 0.33 min | A100    |
| 5    | gpt2-small (124M)  | 预训练     | last                  | all             | 最长训练样本 (120)                                  | 99.62%     | 96.64%    | 96.67%     | 0.69 min | A100    |
| 6    | gpt2-medium (355M) | 预训练     | last                  | last_block      | 最长训练样本 (120)                                  | 87.50%     | 91.28%    | 84.67%     | 0.75 min | A100    |
| 7    | gpt2-large (774M)  | 预训练     | last                  | last_block      | 最长训练样本 (120)                                  | 99.52%     | 98.66%    | 96.67%     | 1.50 min | A100    |
| 8    | gpt2-xl (1558M)    | 预训练     | last                  | last_block      | 最长训练样本 (120)                                  | 99.81%     | 99.81%    | 98.33%     | 2.83 min | A100    |
| 9    | gpt2-xl (1558M)    | 预训练     | last                  | all             | 最长训练样本 (120)                                  | 100.00%    | 98.66%    | 98.67%     | 8.12 min | A100    |
| 10   | gpt2-small (124M)  | 随机       | last                  | all             | 最长训练样本 (120)                                  | 100.00%    | 96.64%    | 93.67%     | 0.69 min | A100    |
| 11   | gpt2-small (124M)  | 预训练     | last                  | LoRA            | 最长训练样本 (120)                                  | 100.00%    | 97.32%    | 96.67%     | 0.75 min | A100    |
| 12   | gpt2-xl (1558M)    | 预训练     | last                  | LoRA            | 最长训练样本 (120)                                  | 100.00%    | 98.66%    | 98.33%     | 5.79 min | A100    |
| 13   | gpt2-small (124M)  | 预训练     | last                  | last_block      | 上下文长度 (1024)                                   | 83.08%     | 87.92%    | 78.33%     | 2.46 min | A100    |
| 14   | gpt2-small (124M)  | 预训练     | last                  | last_block      | 可变: 无填充 (batch size 1)                        | 100.00%    | 98.66%    | 98.00%     | 1.75 min | A100    |
| 15   | gpt2-small (124M)  | 预训练     | last                  | last_block      | 可变: 无填充 (batch size 8)                        | 99.33%     | 98.66%    | 98.33%     | 1.70 min | A100    |
| 16   | gpt2-small (124M)  | 预训练     | last                  | last_block      | 灵活 (最后非填充 token 位置)                        | 99.42%     | 98.66%    | 98.33%     | 0.30 min | A100    |
| 17   | gpt2-small (124M)  | 预训练     | last                  | last_block      | 最长训练样本 (120); 无 causal mask                   | 99.23%     | 98.66%    | 95.33%     | 0.29 min | A100    |
| 18   | gpt2-small (124M)  | 预训练     | last                  | last_block      | 最长训练样本 (120) + `ignore_index` 填充            | 96.63%     | 99.33%    | 95.00%     | 0.28 min | A100    |
| 19   | gpt2-small (124M)  | 预训练     | last + pooled embeddings | last_block   | 最长训练样本 (120)                                  | 97.79%     | 99.33%    | 96.33%     | 0.32 min | A100    |

---

### 使用方式（Usage）

可以使用如下命令复现实验：

- 行 1: `python additional_experiments.py`
- 行 2: `python additional_experiments.py --trainable_token_pos first`
- 行 3: `python additional_experiments.py --trainable_layers last_layer`
- 行 4: `python additional_experiments.py --trainable_layers last_two_blocks`
- 行 5: `python additional_experiments.py --trainable_layers all`
- 行 6: `python additional_experiments.py --model_size "gpt2-medium (355M)"`
- 行 7: `python additional_experiments.py --model_size "gpt2-large (774M)"`
- 行 8: `python additional_experiments.py --model_size "gpt2-xl (1558M)"`
- 行 9: `python additional_experiments.py --model_size "gpt2-xl (1558M)" --trainable_layers all`
- 行 10: `python additional_experiments.py --weights random --trainable_layers all`
- 行 11: `python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 16`
- 行 12: `python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 8 --model_size "gpt2-xl (1558M)"`
- 行 13: `python additional_experiments.py --context_length "model_context_length"`
- 行 14: `python additional_experiments.py --no_padding --batch_size 1`
- 行 15: `python additional_experiments.py --no_padding --batch_size 1 --accumulation_steps 8`
- 行 16: `python additional_experiments.py --trainable_token_pos "flexible"`
- 行 17: `python additional_experiments.py --disable_causal_mask`
- 行 18: `python additional_experiments.py --ignore_index 50256`
- 行 19: `python additional_experiments.py --average_embeddings`

> 注：为了方便实验运行，本实验保持模型和数据集较小，可以在普通笔记本（如 MacBook Air M3）上约 15 分钟完成默认设置训练，无需 GPU。

---

### 实验结果解读（Interpretation）

1. **最后 token vs 第一个 token（行 1 vs 2）**：训练最后一个 token 的性能显著优于训练第一个 token，原因是因果自注意力掩码。
2. **最后 block vs 最后一层（行 1 vs 3）**：训练整个最后 transformer block 比仅训练最后一层效果更好。
3. **最后 block vs 最后两个 block（行 1 vs 4）**：训练最后两个 block 而非仅最后一个 block，准确率提升 3.33%。
4. **最后 block vs 全部层（行 1 vs 5）**：训练所有层仅比训练最后 block 提升 ~2%，训练时间几乎增加三倍，但不如训练最后两个 block。
5. **使用更大预训练模型（行 1 vs 6-8）**：中等模型可能预训练不足或微调配置不适配，5x 与 12x 大模型性能提升明显。
6. **随机权重 vs 预训练权重（行 1,5 vs 10）**：随机权重结果略差 1-3%，显示预训练权重的重要性。
7. **LoRA 微调 vs 全层训练（行 11 vs 5, 行 12 vs 9）**：冻结原模型，仅微调 LoRA 层可提升性能约 1%，且更节省显存。
8. **填充到最大上下文 vs 最长训练样本（行 1 vs 13）**：填充到全上下文长度效果显著下降。
9. **填充 vs 无填充（行 1 vs 14-16）**：无填充或根据最后非填充 token 灵活选择 token 位置，测试准确率略高，但训练时间增加。
10. **禁用因果注意力掩码（行 1 vs 17）**：禁用后所有 token 可互相注意，模型准确率略提升。
11. **忽略 padding 索引（行 1 vs 18）**：对本二分类实验无影响
