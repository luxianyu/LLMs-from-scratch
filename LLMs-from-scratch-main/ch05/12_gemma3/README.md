# 从零开始构建 Gemma 3 270M

本文件夹中的 [standalone-gemma3.ipynb](standalone-gemma3.ipynb) Jupyter 笔记本包含 **Gemma 3 270M** 的从零实现。运行大约需要 **2 GB 内存**。

另一个可选笔记本 [standalone-gemma3-plus-kvcache.ipynb](standalone-gemma3-plus-kvcache.ipynb) 增加了 **KV 缓存**，提升推理性能（但代码复杂度也增加）。  
想了解更多关于 KV 缓存的内容，可参考我的文章：[从零实现 LLM 的 KV 缓存解析与编程](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。

| 模型               | 模式                  | 硬件             | Tokens/sec | GPU 内存 (VRAM) |
| ----------------- | ------------------- | --------------- | ---------- | ----------------- |
| Gemma3Model 270M  | 常规                 | Mac Mini M4 CPU | 8          | -                 |
| Gemma3Model 270M  | 常规编译版           | Mac Mini M4 CPU | 9          | -                 |
| Gemma3Model 270M  | KV 缓存              | Mac Mini M4 CPU | 130        | -                 |
| Gemma3Model 270M  | KV 缓存编译版        | Mac Mini M4 CPU | 224        | -                 |
|                   |                     |                 |            |                   |
| Gemma3Model 270M  | 常规                 | Mac Mini M4 GPU | 16         | -                 |
| Gemma3Model 270M  | 常规编译版           | Mac Mini M4 GPU | Error      | -                 |
| Gemma3Model 270M  | KV 缓存              | Mac Mini M4 GPU | 23         | -                 |
| Gemma3Model 270M  | KV 缓存编译版        | Mac Mini M4 GPU | Error      | -                 |
|                   |                     |                 |            |                   |
| Gemma3Model 270M  | 常规                 | Nvidia A100 GPU | 28         | 1.84 GB           |
| Gemma3Model 270M  | 常规编译版           | Nvidia A100 GPU | 128        | 2.12 GB           |
| Gemma3Model 270M  | KV 缓存              | Nvidia A100 GPU | 26         | 1.77 GB           |
| Gemma3Model 270M  | KV 缓存编译版        | Nvidia A100 GPU | 99         | 2.12 GB           |

---

下图展示了 **Gemma 3 270M** 与 **Qwen3 0.6B** 的并排对比；  
如果你对 Qwen3 0.6B 独立笔记本感兴趣，可以在此查看：[链接](../11_qwen3)  

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gemma3/gemma3-vs-qwen3.webp">

<br>

如需了解更多关于架构差异及与其他模型的比较，请参阅我的文章：[大型 LLM 架构对比：从 DeepSeek-V3 到 Kimi K2 的现代架构设计分析](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)。
