# 构建大型语言模型（从零开始）

<table style="width:100%">
<tr>
<td style="vertical-align:middle; text-align:left;">
<font size="2">
<a href="http://mng.bz/orYv">《从零开始构建大型语言模型》</a>作者为 <a href="https://sebastianraschka.com">Sebastian Raschka</a><br>
<br>中文翻译和代码详细注释由Lux整理，Github下载地址：<a href="https://github.com/luxianyu">https://github.com/luxianyu</a>
    
<br>Lux的Github上还有吴恩达深度学习Pytorch版学习笔记及中文详细注释的代码下载
    
</font>
</td>
<td style="vertical-align:middle; text-align:left;">
<a href="http://mng.bz/orYv"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover-small.webp" width="100px"></a>
</td>
</tr>
</table>

<br>

在 [*Build a Large Language Model (From Scratch)*](http://mng.bz/orYv) 中，你将通过从零实现 LLM 的方式，从内部深入理解大型语言模型的工作原理。书中每个步骤都配有清晰的文字说明、示意图和示例代码，引导你逐步构建自己的 LLM。

本书所描述的方法适用于教育目的下训练小型可用模型，并模拟了构建 ChatGPT 等大规模基础模型的流程。此外，书中还提供了加载大型预训练模型权重以进行微调的代码。


# 目录


<br>

| 章节标题                                                    | 主要代码（快速访问）                                                                                                       | 所有代码 + 补充资料         |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|----------------------------|
| [设置建议](setup)                                           | -                                                                                                                         | -                          |
| 第1章：理解大型语言模型                                     | 无代码                                                                                                                     | -                          |
| 第2章：文本数据处理                                         | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb)（摘要）<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb) | [./ch02](./ch02)           |
| 第3章：实现注意力机制                                       | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb)（摘要）<br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb) | [./ch03](./ch03)           |
| 第4章：从零实现 GPT 模型                                    | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py)（摘要）<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04)           |
| 第5章：无标签数据预训练                                     | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py)（摘要）<br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py)（摘要）<br/>- [exercise-solutions.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb) | [./ch05](./ch05)           |
| 第6章：文本分类微调                                         | - [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb)<br/>- [gpt_class_finetune.py](ch06/01_main-chapter-code/gpt_class_finetune.py)<br/>- [exercise-solutions.ipynb](ch06/01_main-chapter-code/exercise-solutions.ipynb) | [./ch06](./ch06)           |
| 第7章：指令微调                                             | - [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)<br/>- [gpt_instruction_finetuning.py](ch07/01_main-chapter-code/gpt_instruction_finetuning.py)（摘要）<br/>- [ollama_evaluate.py](ch07/01_main-chapter-code/ollama_evaluate.py)（摘要）<br/>- [exercise-solutions.ipynb](ch07/01_main-chapter-code/exercise-solutions.ipynb) | [./ch07](./ch07)           |
| 附录A：PyTorch 入门                                        | - [code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb) | [./appendix-A](./appendix-A) |
| 附录B：参考资料与扩展阅读                                   | 无代码                                                                                                                     | [./appendix-B](./appendix-B) |
| 附录C：习题答案                                           | - [习题答案列表](appendix-C)                                                                                               | [./appendix-C](./appendix-C) |
| 附录D：训练循环的增强                                      | - [appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb)                                                    | [./appendix-D](./appendix-D) |
| 附录E：LoRA 参数高效微调                                    | - [appendix-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb)                                                    | [./appendix-E](./appendix-E) |

<br>
&nbsp;

下面的思维模型总结了本书所涵盖的内容。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">

<br>
&nbsp;

## 前置条件

最重要的前置条件是扎实的 Python 编程基础。  
拥有这些知识后，您将能够顺利探索大型语言模型的精彩世界，并理解书中提供的概念和代码示例。

如果您有一定的深度神经网络经验，可能会对某些概念更为熟悉，因为 LLM 是基于这些架构构建的。

本书使用 PyTorch 从零实现代码，而不依赖任何外部 LLM 库。虽然精通 PyTorch 不是前置条件，但熟悉 PyTorch 基础肯定有帮助。如果您是 PyTorch 新手，附录 A 提供了简明的 PyTorch 入门介绍。或者，您也可以参考我的书 [PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs](https://sebastianraschka.com/teaching/pytorch-1h/) 学习基础知识。

<br>
&nbsp;

## 硬件要求

本书主章节的代码设计为在常规笔记本电脑上可以在合理时间内运行，无需专用硬件。这保证了更多读者可以参与学习。此外，代码会自动使用可用 GPU。（更多建议请参见 [setup](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md) 文档）

&nbsp;
## 视频课程

[17 小时 15 分钟伴随视频课程](https://www.manning.com/livevideo/master-and-build-large-language-models) 对应书中每章代码实现。课程按章节和小节组织，与书籍结构一致，可作为书籍的独立替代或补充跟随资源。

<a href="https://www.manning.com/livevideo/master-and-build-large-language-models"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/video-screenshot.webp?123" width="350px"></a>

&nbsp;

## 配套书 / 后续书籍

[*Build A Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) 是独立书籍，但可视作 *Build A Large Language Model (From Scratch)* 的续集。

该书从预训练模型开始，实施不同推理方法，包括推理时扩展、强化学习和蒸馏，以提升模型推理能力。  

类似于 *Build A Large Language Model (From Scratch)*，[ *Build A Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) 也是从零动手实现这些方法。

<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover.webp?123" width="120px"></a>

- 亚马逊链接（待定）
- [Manning 链接](https://mng.bz/lZ5B)
- [GitHub 仓库](https://github.com/rasbt/reasoning-from-scratch)

<br>

&nbsp;
## 习题

每章都包含若干习题。答案汇总在附录 C，对应的代码笔记本在本仓库主章节文件夹中可获取（例如 [./ch02/01_main-chapter-code/exercise-solutions.ipynb](./ch02/01_main-chapter-code/exercise-solutions.ipynb)）。

除了代码习题，您还可以免费下载一本 170 页的 PDF《[Test Yourself On Build a Large Language Model (From Scratch)](https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch)》，每章包含约 30 道测验题及答案，用于检验学习效果。

<a href="https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/test-yourself-cover.jpg?123" width="150px"></a>

&nbsp;
## 附加材料

一些文件夹提供额外可选资料，供感兴趣的读者使用：

- **设置**
  - [Python 设置技巧](setup/01_optional-python-setup-preferences)
  - [本书使用的 Python 包安装指南](setup/02_installing-python-libraries)
  - [Docker 环境设置指南](setup/03_optional-docker-environment)
- **第2章：文本数据处理**
  - [从零实现 Byte Pair Encoding (BPE) 分词器](ch02/05_bpe-from-scratch/bpe-from-scratch-simple.ipynb)
  - [各种 BPE 实现比较](ch02/02_bonus_bytepair-encoder)
  - [理解 Embedding 层与线性层的区别](ch02/03_bonus_embedding-vs-matmul)
  - [简单数字下的 Dataloader 直觉](ch02/04_bonus_dataloader-intuition)
- **第3章：注意力机制实现**
  - [高效多头注意力实现比较](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
  - [理解 PyTorch Buffers](ch03/03_understanding-buffers/understanding-buffers.ipynb)
- **第4章：从零实现 GPT 模型**
  - [FLOPS 分析](ch04/02_performance-analysis/flops-analysis.ipynb)
  - [KV 缓存](ch04/03_kv-cache)
  - [注意力替代方案](ch04/#attention-alternatives)
    - [分组查询注意力](ch04/04_gqa)
    - [多头潜在注意力](ch04/05_mla)
    - [滑动窗口注意力](ch04/06_swa)
- **第5章：无标签数据预训练**
  - [替代权重加载方法](ch05/02_alternative_weight_loading/)
  - [在 Gutenberg 数据集上预训练 GPT](ch05/03_bonus_pretraining_on_gutenberg)
  - [训练循环增强技巧](ch05/04_learning_rate_schedulers)
  - [预训练超参数优化](ch05/05_bonus_hparam_tuning)
  - [构建与预训练 LLM 交互的用户界面](ch05/06_user_interface)
  - [将 GPT 转换为 Llama](ch05/07_gpt_to_llama)
  - [从零实现 Llama 3.2](ch05/07_gpt_to_llama/standalone-llama32.ipynb)
  - [从零实现 Qwen3 Dense 与 MoE](ch05/11_qwen3/)
  - [从零实现 Gemma 3](ch05/12_gemma3/)
  - [内存高效模型权重加载](ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
  - [为 Tiktoken BPE 分词器扩展新 Token](ch05/09_extending-tokenizers/extend-tiktoken.ipynb)
  - [PyTorch 性能优化，加速 LLM 训练](ch05/10_llm-training-speed)
- **第6章：分类微调**
  - [微调不同层及大模型的额外实验](ch06/02_bonus_additional-experiments)
  - [在 50k IMDb 电影评论数据集上微调不同模型](ch06/03_bonus_imdb-classification)
  - [构建 GPT 垃圾邮件分类器交互界面](ch06/04_user_interface)
- **第7章：指令微调**
  - [寻找近重复样本与被动语态生成的工具](ch07/02_dataset-utilities)
  - [使用 OpenAI API 与 Ollama 评估指令响应](ch07/03_model-evaluation)
  - [生成指令微调数据集](ch07/05_dataset-generation/llama3-ollama.ipynb)
  - [改进指令微调数据集](ch07/05_dataset-generation/reflection-gpt4.ipynb)
  - [使用 Llama 3.1 70B 与 Ollama 生成偏好数据集](ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb)
  - [LLM 对齐的直接偏好优化（DPO）](ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
  - [构建指令微调 GPT 交互界面](ch07/06_user_interface)

更多附加材料请参考 [从零推理仓库](https://github.com/rasbt/reasoning-from-scratch)：

- **Qwen3（从零）基础**
  - [Qwen3 源码讲解](https://github.com/rasbt/reasoning-from-scratch/blob/main/chC/01_main-chapter-code/chC_main.ipynb)
  - [优化后的 Qwen3](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch02/03_optimized-LLM)
- **评估**
  - [基于验证器的评估（MATH-500）](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch03)
  - [多选题评估（MMLU）](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/02_mmlu)
  - [LLM 排行榜评估](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/03_leaderboards)
  - [LLM 作为裁判评估](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/04_llm-judge)

<br>

&nbsp;
## 引用

如果您在研究中使用了本书或代码，请考虑引用。

芝加哥格式引用：

> Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.

BibTeX 条目：


```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```
