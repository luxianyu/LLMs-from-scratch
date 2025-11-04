# 第五章：无标签数据的预训练（Pretraining on Unlabeled Data）

&nbsp;
## 本章主要代码

- [01_main-chapter-code](01_main-chapter-code) 包含本章的主要代码。

&nbsp;
## 额外资源（Bonus Materials）

- [02_alternative_weight_loading](02_alternative_weight_loading) 提供在 OpenAI 模型权重不可用时，从其他来源加载 GPT 模型权重的代码。
- [03_bonus_pretraining_on_gutenberg](03_bonus_pretraining_on_gutenberg) 提供在 Project Gutenberg 全部书籍语料上进一步预训练 LLM 的代码。
- [04_learning_rate_schedulers](04_learning_rate_schedulers) 实现了更复杂的训练函数，包括学习率调度器和梯度裁剪。
- [05_bonus_hparam_tuning](05_bonus_hparam_tuning) 提供可选的超参数调优脚本。
- [06_user_interface](06_user_interface) 实现了一个交互式用户界面，用于与预训练 LLM 进行交互。
- [07_gpt_to_llama](07_gpt_to_llama) 提供将 GPT 架构实现逐步转换为 Llama 3.2 并加载 Meta AI 预训练权重的指南。
- [08_memory_efficient_weight_loading](08_memory_efficient_weight_loading) 展示了如何通过 PyTorch 的 `load_state_dict` 方法更高效地加载模型权重。
- [09_extending-tokenizers](09_extending-tokenizers) 从零实现了 GPT-2 的 BPE 分词器。
- [10_llm-training-speed](10_llm-training-speed) 提供 PyTorch 性能优化技巧，以提高 LLM 训练速度。
- [11_qwen3](11_qwen3) 从零实现 Qwen3 0.6B 和 Qwen3 30B-A3B（Mixture-of-Experts），包括加载基础模型、推理模型和编码模型预训练权重的代码。
- [12_gemma3](12_gemma3) 从零实现 Gemma 3 270M 及 KV 缓存版本，包括加载预训练权重的代码。

<br>
<br>

[![点击观看视频](https://img.youtube.com/vi/Zar2TJv-sE0/0.jpg)](https://www.youtube.com/watch?v=Zar2TJv-sE0)
