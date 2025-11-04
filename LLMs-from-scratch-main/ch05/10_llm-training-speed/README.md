# PyTorch 性能优化技巧以加速 LLM 训练

请注意，本书是以教育为目的编写的，这意味着原始代码刻意保持简单。这是为了提高可读性，并确保在不同硬件（包括 CPU 和 GPU）上的兼容性。然而，你可能会对一些更高级的 PyTorch 和 GPU 功能感兴趣，以提升 LLM 训练的性能。

此文件夹包含三个代码文件，展示了针对 LLM 及第 5 章中引入的训练函数的性能优化：

1. [`00_orig.py`](00_orig.py)：CPU 和单 GPU 训练的第 5 章原始代码。  
   ➤ 运行方式：`python 00_orig.py`

2. [`01_opt_single_gpu.py`](01_opt_single_gpu.py)：单 GPU 训练的优化版本。  
   ➤ 运行方式：`python 01_opt_single_gpu.py`

3. [`02_opt_multi_gpu_ddp.py`](02_opt_multi_gpu_ddp.py)：使用分布式数据并行（DDP）的多 GPU 训练优化版本。  
   ➤ 运行方式：`torchrun --nproc_per_node=4 02_opt_multi_gpu_ddp.py`  
   (**注意：** 为了与 `01_opt_single_gpu.py` 相比尽量保持改动最小，此脚本仅支持通过上述 `torchrun` 进行多进程。这意味着通过 `python 02_opt_multi_gpu_ddp.py` **不** 支持多 GPU。)

**请注意，这些修改将训练速度从每秒 12,525 个 token（单 A100）提升到每秒 142,156 个 token（单 A100），以及每秒 419,259 个 token（4x A100）。**

我计划在未来撰写更详细的差异分析。现在，查看代码改进最简单的方法是在 Visual Studio Code 中打开这些文件，并使用“Compare Selected”功能查看差异。

![VS compare](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/llm-training-speed/vs-code-compare.png)

![PyTorch Tips](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/pytorch-tips/pytorch-tips.webp?1)

&nbsp;
## 单 GPU 速度对比

如上所述，我计划在未来更详细地说明这些改动。现在，本节提供每项修改的 tokens/秒 简单性能概览。所有实验均在 A100 GPU 上运行。

&nbsp;
### 基线

请注意，`00_orig.py` 作为基线文件，未做显著修改，使用了第 5 章的原始代码，除了以下几点：

- 上下文长度扩大 4 倍（解释了 `00_orig.py` 相较第 5 章内存占用较大的原因）；  
- 批量大小增加 4 倍（也是 `00_orig.py` 内存占用较大的原因）；  
- 使用了更大的公有领域书籍，以增加训练数据量。

超参数并未针对最小化损失和减少过拟合进行优化，LLM 最终生成的文本可能不够复杂；然而，这并不影响主要结论，即 `tok/sec` 指标作为速度参考（数值越高越好）。

```bash
ubuntu@159-13-52-60:~$ python 00_orig.py
PyTorch version: 2.6.0+cu124
Using cuda
CUDA version: 12.4

Ep 1, Step 000000, Train: 9.535, Val: 9.609, Step tok/sec: 7238, Avg tok/sec: 0
Ep 1, Step 000015, Train: 6.201, Val: 6.152, Step tok/sec: 12545, Avg tok/sec: 12545
Ep 1, Step 000030, Train: 5.663, Val: 5.688, Step tok/sec: 12490, Avg tok/sec: 12517
Ep 1, Step 000045, Train: 5.316, Val: 5.362, Step tok/sec: 12541, Avg tok/sec: 12525
Every effort moves you, and's, and I am not be a

...

Ep 15, Step 000735, Train: 0.227, Val: 6.818, Step tok/sec: 11599, Avg tok/sec: 12248
Ep 15, Step 000750, Train: 0.300, Val: 6.895, Step tok/sec: 12530, Avg tok/sec: 12253
Ep 15, Step 000765, Train: 0.150, Val: 6.914, Step tok/sec: 12532, Avg tok/sec: 12259
Every effort moves you like best to think which he held in the room in him, the interest was the night, the realities of the affairs Bulstrode's duty, now!' the fact is another man, conquests

Allocated memory: 2.5069 GB
Reserved memory: 26.2617 GB
```

请注意，`01_opt_single_gpu.py` 包含下面按顺序列出的所有修改。  

比较始终基于上一节中第一轮训练后的平均 tok/sec 和已分配内存。

&nbsp;
### 1. 动态创建因果掩码

- 不再保存因果掩码，而是动态创建以减少内存使用（在这里影响很小，但在长上下文模型中，如支持 131k 输入 token 的 Llama 3.2，效果会累积）

之前：
- `Avg tok/sec: 12525`
- `Reserved memory: 26.2617 GB`

之后：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 2. 使用张量核心（Tensor Cores）

- 利用张量核心（仅适用于 A100 等 Ampere 架构及更新的 GPU）

之前：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 3. Fused AdamW 优化器

- 通过设置 `fused=True` 使用 AdamW 的融合内核

之前：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 4. 数据加载器使用固定内存（Pinned Memory）

- 在数据加载器中使用 `pin_memory=True`，预分配并重复利用 GPU 内存

之前：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 5. 使用 bfloat16 精度

- 将 32 位浮点切换为 16 位脑浮点（bfloat16）精度（关于此主题的更多信息，请参见我的[文章](https://magazine.sebastianraschka.com/p/the-missing-bits-llama-2-weights)）

之前：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

&nbsp;
### 6. 使用 PyTorch 原生类替换手写实现

- 用 PyTorch 的原生实现替换了手写的 LayerNorm 和 GeLU

之前：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

之后：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

&nbsp;
### 7. 使用 FlashAttention

- 使用 PyTorch 自带的 FlashAttention 自注意力函数，替换我们的手写多头注意力实现

之前：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

之后：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

&nbsp;
### 8. 使用 `pytorch.compile`

- 使用 `torch.compile(model)`。注意首次迭代总是较慢，之后才会加速。由于 `Avg tok/sec` 测量只包括平均计算的第一行，因此我们现在使用第一轮结束时的 `Step tok/sec`。

之前：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

之后：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

<br>

---

**Windows 注意事项**

- 在 Windows 上编译可能比较棘手  
- `torch.compile()` 使用 Inductor，它 JIT 编译内核，需要工作正常的 C/C++ 工具链  
- 对于 CUDA，Inductor 还依赖 Triton，可通过社区包 `triton-windows` 获取  
  - 如果出现 `cl not found`，请[安装带 C++ 工作负载的 Visual Studio Build Tools](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170)，并从 “x64 Native Tools” 提示符运行 Python  
  - 如果出现 CUDA 下 `triton not found`，请安装 `triton-windows`（例如，`uv pip install "triton-windows<3.4"`）  
- 对于 CPU，有读者建议遵循[Windows 上 PyTorch Inductor 指南](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)  
  - 安装 Visual Studio 2022 时务必安装英文语言包，以避免 UTF-8 错误  
  - 请注意，代码需要通过 “Visual Studio 2022 Developer Command Prompt” 运行，而不是 Notebook  
- 如果设置过于复杂，可以跳过编译；**编译是可选的，所有代码示例在不编译情况下也能正常运行**

---

&nbsp;
### 9. 词汇表填充（Vocabulary Padding）

- 此处，我们将词汇表稍微从 50,257 增加到 50,304，这是最接近的 64 倍数。该技巧由我的前同事 Carlos Mocholi 建议，他提到这最初来自 Andrej Karpathy（可能来自[此帖子](https://x.com/karpathy/status/1621578354024677377)）。Karpathy 的建议基于与 PyTorch 团队的交流，他们就 `torch.compile` 提供了建议，如 [Bertrand Maher](https://www.linkedin.com/feed/update/urn:li:activity:7309569006057795584?commentUrn=urn%3Ali%3Acomment%3A%28activity%3A7309569006057795584%2C7309754284185669632%29&dashCommentUrn=urn%3Ali%3Afsd_comment%3A%287309754284185669632%2Curn%3Ali%3Aactivity%3A7309569006057795584%29) 所述。  
- 一个很好的资源是 [NVIDIA 关于张量形状的指南](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape)，通常批量大小和线性层维度会选为某些值的倍数。  
- 此外，词汇表填充技巧很早就被 NVIDIA 的 Megatron 团队描述过（参见 2019 年 [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) 论文）。

之前：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

之后：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

&nbsp;
### 10. 增加批量大小

- 最后，将批量大小增加到 GPU 支持的最大 2 的幂

之前：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

之后：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

&nbsp;
## 多 GPU 速度对比

这可能不是完全公平的对比，因为我们现在使用 4 个 GPU 而非 1 个，但使用分布式数据并行（Distributed Data Parallel），在训练不受限于 GPU 内存的情况下，是最快的多 GPU 技术，当然可以显著加快速度：

之前（单 GPU）：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

之后（4 个 GPU）：
- `Step tok/sec: 419259`
- `Reserved memory: 22.7969 GB`
