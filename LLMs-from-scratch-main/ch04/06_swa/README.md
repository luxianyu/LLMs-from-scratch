# 滑动窗口注意力（Sliding Window Attention, SWA）

本附加内容展示了在使用滑动窗口注意力（SWA）替代常规多头注意力（MHA）时的内存节省效果。

&nbsp;
## 介绍

什么是滑动窗口注意力（SWA）？  
如果我们将常规自注意力视为一种*全局*注意力机制，因为每个序列元素都可以访问序列中的所有其他元素，那么我们可以将 SWA 看作*局部*注意力，因为在这里我们限制了当前查询位置周围的上下文大小。下图进行了说明。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/1.webp?2" alt="Sliding Window Attention" width="500px" />

如上图所示，每个 token 并不是关注所有之前的 token，而只关注其位置附近的固定大小局部窗口。这种局部化的注意力大幅降低了 KV 缓存的大小。

在本介绍的剩余部分，我们将讨论在 [Gemma 3](https://arxiv.org/abs/2503.19786) 中的 SWA，该模型在 [../../ch05/12_gemma3](../../ch05/12_gemma3) 中从零实现。

滑动窗口注意力最初在 [2020 年的 LongFormer 论文](https://arxiv.org/abs/2004.05150) 中提出，但我们关注 Google 的 Gemma 模型的原因是，它们是非常好的开源权重模型，表明滑动窗口注意力在近期高性能模型中确实是一种可行的方法。

[Gemma 2](https://arxiv.org/abs/2408.00118) 使用了一种混合方法，将局部（滑动窗口）和全局注意力层以 1:1 的比例结合。每个 token 可以关注长度为 4k 的上下文窗口。之所以采用这种 1:1 的混合方式，是因为它在效率与全局上下文建模之间取得了平衡，因为仅使用局部注意力的 LLM 可能过于受限。

[Gemma 3](https://arxiv.org/abs/2503.19786) 则进一步优化了效率。它在滑动窗口层和全局注意力层之间采用了 5:1 的比例，也就是说，每五层局部注意力层就有一层全局层。此外，滑动窗口大小从 Gemma 2 的 4096 token 缩小到 Gemma 3 的 1024 token。

有趣的是，Gemma 3 技术报告中的消融研究表明，这些改变对整体模型质量的影响很小。换句话说，通过滑动窗口注意力实现的显著内存和计算节省，只会带来极小的建模性能损失。

&nbsp;
## 滑动窗口注意力（SWA）内存节省

内存节省主要体现在 KV 存储上。我们可以用以下公式计算 KV 存储大小：


bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads


使用 SWA 时，上述公式中的序列长度（seqlen）被窗口大小 W 替代。因此，使用滑动窗口注意力时，KV 缓存大小可减少一个 "W / seqlen" 的因子。（注意，为简单起见，这假设每一层都使用滑动窗口注意力。）

你可以使用此文件夹中的 [memory_estimator_swa.py](memory_estimator_swa.py) 脚本，将其应用于不同的模型配置，以查看使用 SWA 替代 MHA 可以节省多少内存。


```bash
➜ uv run memory_estimator_swa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 1024 --swa_ratio "5:1"
==== Config ====
context_length         : 32768
sliding_window_size    : 1024
emb_dim                : 4096
n_heads                : 32
n_layers               : 32
n_kv_groups            : 4
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 128
GQA n_kv_heads         : 8
Effective SWA window W : 1024
Layer ratio (SWA:Full) : 5:1
Distributed layers     : 27 SWA, 5 FULL

==== KV-cache totals across all layers ====
MHA KV total           : 17.18 GB
GQA KV total           : 4.29 GB
MHA + SWA (Ratio: 5:1) : 3.14 GB
MHA + GQA (Ratio: 5:1) : 0.78 GB
```

请注意，Gemma 3 在使用 SWA 时是与 GQA 结合的。

下图展示了在不同上下文长度下，使用 SWA 相对于 MHA 的内存节省效果：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/4.webp?2" alt="SWA" width="=800px" />

&nbsp;

你可以通过以下命令复现这些图表：


```bash
plot_memory_estimates_swa.py \
  --emb_dim 4096 --n_heads 48 --n_layers 36 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 2048 --swa_ratio "5:1"
```


&nbsp;
## SWA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_swa.py](gpt_with_kv_swa.py) 脚本提供了在 GPT 模型实现背景下，用于比较 MHA 和 SWA 内存使用的实践示例。

请注意，SWA 也可以与 MLA 和 GQA 结合使用（如前所述），但为简化起见，这里没有这样做。

还请注意，该模型未经过训练，因此会生成无意义的文本。  
不过，你可以将其作为第 5–7 章中标准 GPT 模型的直接替代版本进行训练。

最后，该实现使用了在[另一个附加章节](../03_kv-cache)中解释的 KV 缓存，因此内存节省更加明显。


```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_swa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--sliding_window_size 1024 \
--sliding_window_stride 5   # like Gemma 3

...

Time: 514.38 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

我们之所以没有看到像上图那样显著的节省，有两个原因：

1. 我使用了较小的配置，以便模型能够在合理的时间内完成生成。
2. 更重要的是，这里我们观察的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（但这是一个需要单独分析的主题）。
