# Qwen3 从零开始

本文件夹中的 [standalone-qwen3.ipynb](standalone-qwen3.ipynb) Jupyter 笔记本包含 Qwen3 0.6B、1.7B、4B、8B 和 32B 的从零实现。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">

本文件夹中的 [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb) 和 [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb) 笔记本包含 30B-A3B Mixture-of-Experts（MoE）的从零实现，包括 Thinking、Instruct 和 Coder 模型变体。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-coder-flash-overview.webp?123" width="430px">

&nbsp;
# Qwen3 从零实现代码

本文件夹中的独立笔记本包含线性顺序的从零实现代码：

1. [standalone-qwen3.ipynb](standalone-qwen3.ipynb)：基础 Qwen3 模型，无额外功能  
2. [standalone-qwen3-plus-kvcache.ipynb](standalone-qwen3-plus-kvcache.ipynb)：与上面相同，但增加 KV 缓存以提高推理效率  
3. [standalone-qwen3-moe.ipynb](standalone-qwen3-moe.ipynb)：与第一个笔记本类似，但为 Mixture-of-Experts（MoE）变体  
4. [standalone-qwen3-moe-plus-kvcache.ipynb](standalone-qwen3-moe-plus-kvcache.ipynb)：与上面相同，但增加 KV 缓存以提高推理效率  

此外，我还将代码整理成一个 Python 包 [这里](../../pkg/llms_from_scratch/)（包含单元测试和 CI），可以按下文所述方式运行。

&nbsp;
# 训练

`Qwen3Model` 类的实现风格类似于 `GPTModel` 类，因此可以在第 5 章训练和第 6、7 章微调中作为直接替换使用。

&nbsp;
# 通过 `llms-from-scratch` 包使用 Qwen3

为了更方便地使用 Qwen3 的从零实现，你也可以使用基于本仓库源码的 PyPI 包 `llms-from-scratch`，路径为 [pkg/llms_from_scratch](../../pkg/llms_from_scratch)。

&nbsp;
#### 1) 安装


```bash
pip install llms_from_scratch tokenizers
```

&nbsp;
#### 2) 模型及文本生成设置

指定要使用的模型：


```python
USE_REASONING_MODEL = True
# Uses the base model if USE_REASONING_MODEL = False

USE_INSTRUCT_MODEL = False
# Uses the instruct mode (without reasoning) if 
# USE_REASONING_MODEL = True
# USE_INSTRUCT_MODEL = True
# This setting does have no effect if USE_REASONING_MODEL = False


# Use
# USE_REASONING_MODEL = True
# For Qwen3 Coder Flash model as well
```

用户可以自定义的基础文本生成设置。使用 150 个 token 时，0.6B 模型大约需要 1.5 GB 内存。


```python
MAX_NEW_TOKENS = 150
TEMPERATURE = 0.
TOP_K = 1
```

&nbsp;
#### 3a) 0.6B 模型的权重下载与加载

以下内容会根据上方选择的模型类型（reasoning 或 base）自动下载权重文件。  
请注意，本节仅针对 0.6B 模型。如果你想使用更大规模的模型（1.7B、4B、8B 或 32B），可以跳过本节，直接阅读 3b) 节。

```python
from llms_from_scratch.qwen3 import download_from_huggingface

repo_id = "rasbt/qwen3-from-scratch"

if USE_REASONING_MODEL:
    filename = "qwen3-0.6B.pth"
    local_dir = "Qwen3-0.6B"    
else:
    filename = "qwen3-0.6B-base.pth"   
    local_dir = "Qwen3-0.6B-Base"

download_from_huggingface(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir
)
```

随后，模型权重将按如下方式加载：


```python
from pathlib import Path
import torch

from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B

model_file = Path(local_dir) / filename

model = Qwen3Model(QWEN_CONFIG_06_B)
model.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)
model.to(device);
```

&nbsp;
#### 3b) 更大规模 Qwen 模型的权重下载与加载

如果你想使用更大规模的 Qwen 模型，例如 1.7B、4B、8B 或 32B，请使用下方代码替代 3a) 节中的代码。  
注意：此部分需要额外的代码依赖。

```bash
pip install safetensors huggingface_hub
```

Then use the following code (make appropriate changes to `USE_MODEL` to select the desired model size)

```python
USE_MODEL = "1.7B"

if USE_MODEL == "1.7B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_1_7B as QWEN3_CONFIG
elif USE_MODEL == "4B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_4B as QWEN3_CONFIG
elif USE_MODEL == "8B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_8B as QWEN3_CONFIG
elif USE_MODEL == "14B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_14B as QWEN3_CONFIG
elif USE_MODEL == "32B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_32B as QWEN3_CONFIG
elif USE_MODEL == "30B-A3B":
    from llms_from_scratch.qwen3 import QWEN3_CONFIG_30B_A3B as QWEN3_CONFIG
else:
    raise ValueError("Invalid USE_MODEL name.")
    
repo_id = f"Qwen/Qwen3-{USE_MODEL}"
local_dir = f"Qwen3-{USE_MODEL}"

if not USE_REASONING_MODEL:
  repo_id = f"{repo_id}-Base"
  local_dir = f"{local_dir}-Base"
```

现在，将权重下载并加载到 `model` 中：


```python
from llms_from_scratch.qwen3 import (
    Qwen3Model,
    download_from_huggingface_from_snapshots,
    load_weights_into_qwen
)

device = (
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cpu")
)

with device:
    model = Qwen3Model(QWEN3_CONFIG)

weights_dict = download_from_huggingface_from_snapshots(
    repo_id=repo_id,
    local_dir=local_dir
)
load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
model.to(device)  # only required for the MoE models
del weights_dict  # delete weight dictionary to free up disk space
```


&nbsp;

#### 4) 初始化分词器

以下代码用于下载并初始化分词器：


```python
from llms_from_scratch.qwen3 import Qwen3Tokenizer

if USE_REASONING_MODEL:
    tok_filename = "tokenizer.json"    
else:
    tok_filename = "tokenizer-base.json"   

tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_file_path,
    repo_id=repo_id,
    apply_chat_template=USE_REASONING_MODEL,
    add_generation_prompt=USE_REASONING_MODEL,
    add_thinking=not USE_INSTRUCT_MODEL
)
```



&nbsp;

#### 5) 文本生成

最后，我们可以通过以下代码生成文本：


```python
prompt = "Give me a short introduction to large language models."
input_token_ids = tokenizer.encode(prompt)
```





```python
from llms_from_scratch.ch05 import generate
import time

torch.manual_seed(123)

start = time.time()

output_token_ids = generate(
    model=model,
    idx=torch.tensor(input_token_ids, device=device).unsqueeze(0),
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
    top_k=1,
    temperature=0.
)

total_time = time.time() - start
print(f"Time: {total_time:.2f} sec")
print(f"{int(len(output_token_ids[0])/total_time)} tokens/sec")

if torch.cuda.is_available():
    max_mem_bytes = torch.cuda.max_memory_allocated()
    max_mem_gb = max_mem_bytes / (1024 ** 3)
    print(f"Max memory allocated: {max_mem_gb:.2f} GB")

output_text = tokenizer.decode(output_token_ids.squeeze(0).tolist())

print("\n\nOutput text:\n\n", output_text + "...")
```
使用 Qwen3 0.6B 推理模型时，输出应类似下方所示（此处运行环境为 A100）：


```
Time: 6.35 sec
25 tokens/sec
Max memory allocated: 1.49 GB


Output text:

 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.<|endoftext|>Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```


对于更大规模的模型，你可能更倾向于使用流式生成版本，该版本会在每个 token 生成后立即打印：


```python
from llms_from_scratch.generate import generate_text_simple_stream

input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

for token in generate_text_simple_stream(
    model=model,
    token_ids=input_token_ids_tensor,
    max_new_tokens=150,
    eos_token_id=tokenizer.eos_token_id
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )
```

```
 <|im_start|>user
Give me a short introduction to large language models.<|im_end|>
Large language models (LLMs) are advanced artificial intelligence systems designed to generate human-like text. They are trained on vast amounts of text data, allowing them to understand and generate coherent, contextually relevant responses. LLMs are used in a variety of applications, including chatbots, virtual assistants, content generation, and more. They are powered by deep learning algorithms and can be fine-tuned for specific tasks, making them versatile tools for a wide range of industries.<|endoftext|>Human resources department of a company is planning to hire 100 new employees. The company has a budget of $100,000 for the recruitment process. The company has a minimum wage of $10 per hour. The company has a total of...
```



&nbsp;

#### 提示 1：通过编译加速推理

为了获得最多 4 倍的加速，请替换


```python
model.to(device)
```

with

```python
model.to(device)
model = torch.compile(model)
```

注意：编译时会产生显著的数分钟初始开销，加速效果将在第一次调用 `generate` 后生效。  

下表展示了在 A100 上连续调用 `generate` 的性能比较：

|                          | 硬件             | Tokens/sec | 内存      |
| ------------------------ | ----------------|----------- | -------- |
| Qwen3Model 0.6B          | Nvidia A100 GPU | 25         | 1.49 GB  |
| Qwen3Model 0.6B 编译版   | Nvidia A100 GPU | 107        | 1.99 GB  |

&nbsp;
#### 提示 2：通过 KV 缓存加速推理

在 CPU 上运行模型时，使用 KV 缓存的 `Qwen3Model` 替代版本可以显著提升推理性能。  
（详情请参考我的文章 [从零实现 LLM 中的 KV 缓存：原理与代码](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) 了解更多 KV 缓存相关内容。）


```python
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple

model = Qwen3Model(QWEN_CONFIG_06_B)
# ...
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(PROMPT, tokenizer).to(device),
    max_new_tokens=MAX_NEW_TOKENS,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

请注意，上表中仅列出了 Nvidia CUDA 设备的峰值内存，因为计算起来较为简单。然而，其他设备的内存使用情况可能类似，因为它们使用类似的精度格式，并且 KV 缓存存储在生成 150-token 文本时会进一步降低内存使用量（不过，不同设备的矩阵乘法实现可能不同，可能导致峰值内存需求有所差异；而对于更长的上下文，KV 缓存的内存可能会显著增加）。

| 模型              | 模式               | 硬件             | Tokens/sec | GPU 内存 (VRAM) |
| ---------------- | ---------------- | ---------------- | ---------- | ----------------- |
| Qwen3Model 0.6B  | 常规               | Mac Mini M4 CPU  | 1          | -                 |
| Qwen3Model 0.6B  | 常规编译版         | Mac Mini M4 CPU  | 1          | -                 |
| Qwen3Model 0.6B  | KV 缓存            | Mac Mini M4 CPU  | 80         | -                 |
| Qwen3Model 0.6B  | KV 缓存编译版      | Mac Mini M4 CPU  | 137        | -                 |
|                  |                   |                  |            |                   |
| Qwen3Model 0.6B  | 常规               | Mac Mini M4 GPU  | 21         | -                 |
| Qwen3Model 0.6B  | 常规编译版         | Mac Mini M4 GPU  | Error      | -                 |
| Qwen3Model 0.6B  | KV 缓存            | Mac Mini M4 GPU  | 28         | -                 |
| Qwen3Model 0.6B  | KV 缓存编译版      | Mac Mini M4 GPU  | Error      | -                 |
|                  |                   |                  |            |                   |
| Qwen3Model 0.6B  | 常规               | Nvidia A100 GPU  | 26         | 1.49 GB           |
| Qwen3Model 0.6B  | 常规编译版         | Nvidia A100 GPU  | 107        | 1.99 GB           |
| Qwen3Model 0.6B  | KV 缓存            | Nvidia A100 GPU  | 25         | 1.47 GB           |
| Qwen3Model 0.6B  | KV 缓存编译版      | Nvidia A100 GPU  | 90         | 1.48 GB           |

请注意，上述所有设置经过测试，生成的文本输出相同。

&nbsp;

#### 提示 3：批量推理

我们可以通过批量推理进一步提高吞吐量。虽然这并非完全可比（因为此时输入序列数量更多），但可以提高每秒生成 token 数，同时以增加内存使用为代价。

这只需要对提示（prompt）的准备进行小幅修改。例如，考虑下面的批量提示：


```python
from llms_from_scratch.ch04 import generate_text_simple
from llms_from_scratch.qwen3 import Qwen3Model, QWEN_CONFIG_06_B
# ...

prompts = [
    "Give me a short introduction to neural networks.",
    "Give me a short introduction to machine learning.",
    "Give me a short introduction to deep learning models.",
    "Give me a short introduction to natural language processing.",
    "Give me a short introduction to generative AI systems.",
    "Give me a short introduction to transformer architectures.",
    "Give me a short introduction to supervised learning methods.",
    "Give me a short introduction to unsupervised learning.",
]

tokenized_prompts = [tokenizer.encode(p) for p in prompts]
max_len = max(len(t) for t in tokenized_prompts)
padded_token_ids = [
    t + [tokenizer.pad_token_id] * (max_len - len(t)) for t in tokenized_prompts
]
input_tensor = torch.tensor(padded_token_ids).to(device)

output_token_ids = generate_text_simple(
    model=model,
    idx=input_tensor,
    max_new_tokens=150,
    context_size=QWEN_CONFIG_06_B["context_length"],
)
```

KV 缓存版本的代码与上述类似，不同之处在于需要使用以下替代版本：


```python
from llms_from_scratch.kv_cache_batched.generate import generate_text_simple
from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model
```


以下实验在批量大小为 8 的情况下进行。

| 模型              | 模式               | 硬件             | 批量大小 | Tokens/sec | GPU 内存 (VRAM) |
| ---------------- | ----------------- | ---------------- | ---------- | ---------- | ----------------- |
| Qwen3Model 0.6B  | 常规               | Mac Mini M4 CPU  | 8          | 2          | -                 |
| Qwen3Model 0.6B  | 常规编译版         | Mac Mini M4 CPU  | 8          | -          | -                 |
| Qwen3Model 0.6B  | KV 缓存            | Mac Mini M4 CPU  | 8          | 92         | -                 |
| Qwen3Model 0.6B  | KV 缓存编译版      | Mac Mini M4 CPU  | 8          | 128        | -                 |
|                  |                   |                  |            |            |                   |
| Qwen3Model 0.6B  | 常规               | Mac Mini M4 GPU  | 8          | 36         | -                 |
| Qwen3Model 0.6B  | 常规编译版         | Mac Mini M4 GPU  | 8          | -          | -                 |
| Qwen3Model 0.6B  | KV 缓存            | Mac Mini M4 GPU  | 8          | 61         | -                 |
| Qwen3Model 0.6B  | KV 缓存编译版      | Mac Mini M4 GPU  | 8          | -          | -                 |
|                  |                   |                  |            |            |                   |
| Qwen3Model 0.6B  | 常规               | Nvidia A100 GPU  | 8          | 184        | 2.19 GB           |
| Qwen3Model 0.6B  | 常规编译版         | Nvidia A100 GPU  | 8          | 351        | 2.19 GB           |
| Qwen3Model 0.6B  | KV 缓存            | Nvidia A100 GPU  | 8          | 140        | 3.13 GB           |
| Qwen3Model 0.6B  | KV 缓存编译版      | Nvidia A100 GPU  | 8          | 280        | 1.75 GB           |


