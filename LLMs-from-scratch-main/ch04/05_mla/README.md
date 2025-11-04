# 多头潜空间注意力（Multi-Head Latent Attention, MLA）

本节附录介绍了多头潜空间注意力（MLA）相较于常规多头注意力（MHA）的**显著内存节省效果**。

&nbsp;
## 一、简介

在 [../04_gqa](../04_gqa) 中，我们讨论了**分组查询注意力（Grouped-Query Attention, GQA）**，它是针对 MHA 的一种计算效率优化方案。根据 [原始 GQA 论文](https://arxiv.org/abs/2305.13245) 和 [Llama 2 论文](https://arxiv.org/abs/2307.09288) 的消融研究，GQA 在模型性能上与 MHA 相当。

而 **多头潜空间注意力（MLA）**（应用于 [DeepSeek V2、V3 和 R1](https://arxiv.org/abs/2412.19437)）采用了**不同的节省内存策略**，尤其适合与 **KV 缓存（KV Cache）** 结合使用。  
与 GQA 共享 K/V 头的方式不同，MLA 将 **Key 和 Value 张量先压缩到一个低维潜空间中**再存储到 KV 缓存中。  
在推理（Inference）阶段，再将这些压缩后的张量通过线性投影还原至原始维度。  
虽然这会额外引入一次矩阵乘法，但显著减少了内存占用。

&nbsp;

![MLA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/1.webp)

&nbsp;

（补充说明：在训练阶段，Query 向量也会被压缩，但推理阶段不会。）

值得注意的是，**MLA 并非 DeepSeek V3 的新特性**，其前身 [DeepSeek V2](https://arxiv.org/abs/2405.04434) 已经使用并首次提出了 MLA。  
V2 论文中还包含若干有趣的消融实验，可以解释为什么 DeepSeek 团队选择 MLA 而非 GQA（如下图所示）。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/2.webp" alt="GQA" width="500px" />

&nbsp;

如上图所示，GQA 的表现略逊于 MHA，而 MLA 不仅节省内存，还**在建模性能上略优于 MHA**。  
这也解释了 DeepSeek 团队选择 MLA 的原因。  
（若论文中能进一步比较 MLA 与 GQA 在“每个 token 的 KV 缓存占用”方面的差异，效果将更直观。）

总而言之，MLA 是一种**巧妙的技巧**：  
在减少 KV 缓存内存占用的同时，甚至在性能上略有提升。

&nbsp;
## 二、MLA 的内存节省分析

MLA 的主要内存节省体现在 **KV 缓存存储** 中。  
KV 缓存的存储大小可以用以下公式估算：

$$
\text{bytes} \approx \text{batch\_size} \times \text{seq\_len} \times \text{n\_layers} \times \text{latent\_dim} \times \text{bytes\_per\_elem}
$$

相比之下，常规 **MHA** 的 KV 缓存存储量为：

$$
\text{bytes} \approx \text{batch\_size} \times \text{seq\_len} \times \text{n\_layers} \times \text{embed\_dim} \times 2 (K,V) \times \text{bytes\_per\_elem}
$$

这意味着，在 MLA 中，我们将  
`embed_dim × 2 (K,V)`  
压缩为  
`latent_dim`。  

换句话说，MLA 仅存储压缩后的潜空间表示，而非完整的 Key 与 Value 向量。

&nbsp;

你可以使用本文件夹中的 [`memory_estimator_mla.py`](memory_estimator_mla.py) 脚本，  
针对不同模型配置，估算使用 MLA 相较于 MHA 时的内存节省比例。

```bash
➜ uv run memory_estimator_mla.py \
  --context_length 8192 \
  --emb_dim 2048 \
  --n_heads 24 \
  --n_layers 48 \
  --n_kv_groups 4 \
  --batch_size 1 \
  --dtype bf16 \
  --latent_dim 1024
==== Config ====
context_length   : 8192
emb_dim          : 2048
n_heads          : 24
n_layers         : 48
n_kv_groups      : 4
latent_dim       : 1024
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 86
GQA n_kv_heads   : 6

==== KV-cache totals across all layers ====
MHA total KV cache  : 3.25 GB
GQA total KV cache  : 0.81 GB
MLA total KV cache  : 0.81 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
Ratio (MHA / MLA)   : 4.03x
Savings (MLA vs MHA): 75.19%
```

请注意，上述压缩（`--emb_dim 2048 -> latent_dim 1024`）旨在实现与 GQA 类似的节省效果。  
在实践中，压缩是一种需要仔细研究的超参数，因为如果将 `latent_dim` 设得过小，可能会对建模性能产生负面影响（类似于在 GQA 中选择过多的 `n_kv_groups`）。

在下图中，针对不同的 `latent_dim` 值，展示了在使用 MLA 相对于 MHA 时的节省效果（作为上下文长度的函数）：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/3.webp?2" alt="GQA" width="500px" />

&nbsp;

你可以通过以下命令复现该图表：

```bash
uv run plot_memory_estimates_mla.py



&nbsp;
## MLA 代码示例

此文件夹中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_mla.py](gpt_with_kv_mla.py) 脚本提供了在 GPT 模型实现背景下，用于比较 MHA 和 MLA 内存使用的实际示例。

这里的 MLA 代码灵感来自以下实现：  
[https://huggingface.co/bird-of-paradise/deepseek-mla](https://huggingface.co/bird-of-paradise/deepseek-mla)

请注意，MLA 也可以与 [GQA](../04_gqa) 结合使用，但为简化起见，这里没有这样做。（目前我也不知道有突出的 LLM 采用这种做法。）

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
uv run gpt_with_kv_mla.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--latent_dim 192 # (768×2)/192 = 8× compression

...

Time: 487.21 sec
67 tokens/sec
Max memory allocated: 0.68 GB
```

我们之所以没有看到像上面图表中那样显著的节省，原因有两个：

1. 我使用了较小的配置，以便模型能够在合理的时间内完成生成。  
2. 更重要的是，这里我们观察的是整个模型，而不仅仅是注意力机制；模型中的全连接层占用了大部分内存（但这是一个需要单独分析的主题）。

