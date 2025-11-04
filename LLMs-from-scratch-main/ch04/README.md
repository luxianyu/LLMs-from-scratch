# 第四章：从零实现 GPT 模型生成文本

&nbsp;
## 本章主要代码

- [01_main-chapter-code](01_main-chapter-code) 包含本章的主要代码。

&nbsp;
## 附加材料

- [02_performance-analysis](02_performance-analysis) 包含可选代码，用于分析本章实现的 GPT 模型的性能
- [03_kv-cache](03_kv-cache) 实现了一个 KV 缓存，用于在推理过程中加速文本生成
- [ch05/07_gpt_to_llama](../ch05/07_gpt_to_llama) 提供了一个逐步指南，将 GPT 架构实现转换为 Llama 3.2，并加载 Meta AI 的预训练权重（完成第四章后，查看替代架构可能很有趣，但你也可以在阅读第五章后再研究）

&nbsp;
## 注意力机制的替代方案

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/attention-alternatives/attention-alternatives.webp">

&nbsp;

- [04_gqa](04_gqa) 介绍了 Grouped-Query Attention (GQA)，大多数现代 LLM（如 Llama 4、gpt-oss、Qwen3、Gemma 3 等）使用它作为常规多头注意力（MHA）的替代方案
- [05_mla](05_mla) 介绍了 Multi-Head Latent Attention (MLA)，DeepSeek V3 使用它作为常规多头注意力（MHA）的替代方案
- [06_swa](06_swa) 介绍了 Sliding Window Attention (SWA)，Gemma 3 等模型使用它

&nbsp;
## 更多内容

在下面的视频中，我提供了一个代码演示课程，涵盖了本章部分内容，作为补充材料。

<br>
<br>

[![视频链接](https://img.youtube.com/vi/YSAkgEarBGE/0.jpg)](https://www.youtube.com/watch?v=YSAkgEarBGE)
