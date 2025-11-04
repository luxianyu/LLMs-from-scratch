# 额外资料：KV 缓存

**该文件夹实现了在 GPT 模型中添加 KV 缓存的功能。**

&nbsp;
## 概览

简而言之，KV 缓存用于存储中间的 key (K) 和 value (V) 计算结果，以便在推理过程中重复使用，从而在生成响应时显著加快速度。缺点是增加了代码复杂度，增加了内存使用量，并且不能在训练期间使用。然而，在部署 LLM 时，推理速度的提升通常非常值得，即使代码和内存开销有所增加。

&nbsp;
## 工作原理

假设 LLM 正在生成一些文本。具体来说，假设 LLM 收到如下提示词："Time flies"。

下图展示了修改自第 3 章的注意力分数计算示意图，其中高亮显示了 key 和 value 向量：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

如我们在第 2 章和第 4 章学习的，LLM 是一次生成一个词（或 token）。假设 LLM 生成了单词 "fast"，下一轮的提示词变为 "Time flies fast"。如下图所示：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

比较前两张图可见，前两个 token 的 key 和 value 向量完全相同，因此在每轮生成下一个 token 时重新计算它们是浪费的。

KV 缓存的思路就是实现一个缓存机制，存储之前生成的 key 和 value 向量以重复使用，从而避免不必要的重复计算。

&nbsp;

## KV 缓存实现

实现 KV 缓存有多种方法，主要思想是每次生成新 token 时只计算新的 key 和 value 张量。

我选择了一个强调代码可读性的简单实现。最直观的方式是直接浏览代码变化，看看实现方法。

该文件夹中有两个文件：

1. [`gpt_ch04.py`](gpt_ch04.py)：从第 3 章和第 4 章独立出来的完整代码，实现 LLM 并运行简单文本生成函数。
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)：与上面相同，但加入了 KV 缓存的必要修改。

你可以：

a. 打开 [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py) 文件，关注标记 `# NEW` 的新修改部分：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b. 使用文件对比工具查看两个代码文件的差异：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

下面是实现细节的简要说明。

&nbsp;

### 1. 注册缓存 buffer

在 `MultiHeadAttention` 构造函数中添加两个非持久性 buffer，`cache_k` 和 `cache_v`，用于保存跨步的 key 和 value 拼接结果：

```python
self.register_buffer("cache_k", None, persistent=False)
self.register_buffer("cache_v", None, persistent=False)

```

&nbsp;

### 2. 带 `use_cache` 标志的前向传播

接着扩展 `MultiHeadAttention` 类的 `forward` 方法，增加 `use_cache` 参数。在将新 token 投影为 `keys_new`、`values_new` 和 `queries` 后，我们初始化 KV 缓存或将新计算结果追加到已有缓存中：


```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
        
    # ...
    
    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

&nbsp;


### 3. 清空缓存

在生成文本时，对于独立序列（例如多次文本生成调用），必须重置两个缓存。因此，我们在 `MultiHeadAttention` 类中添加一个清空缓存的方法：


```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

&nbsp;

### 4. 在完整模型中传递 `use_cache`

在对 `MultiHeadAttention` 类进行修改后，我们现在修改 `GPTModel` 类。首先，我们为 token 索引添加位置跟踪：


```python
self.current_pos = 0
```

然后，我们将原本的一行调用替换为显式循环，并在每个 Transformer 块中传递 `use_cache` 参数：


```python
def forward(self, in_idx, use_cache=False):
    # ...
 
    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,            
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )
    
    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

上述更改还需要对 `TransformerBlock` 类进行小幅修改，以接受 `use_cache` 参数：

```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

最后，我们在 `GPTModel` 中添加一个模型级别的重置方法，以便一次性清空所有 Transformer 块的缓存：


```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

&nbsp;

### 5. 在文本生成中使用缓存

在对 `GPTModel`、`TransformerBlock` 和 `MultiHeadAttention` 做出修改之后，下面展示了如何在一个简单的文本生成函数中使用 KV 缓存：


```python
def generate_text_simple_cached(model, idx, max_new_tokens, 
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

请注意，在步骤 c) 中我们只将新生成的 token 输入模型，通过 `logits = model(next_idx, use_cache=True)`。如果不使用缓存，我们需要将整个输入提供给模型 `logits = model(idx[:, -ctx_len:], use_cache=False)`，因为模型没有存储的 key 和 value 可供重用。

&nbsp;

## 简单性能比较

在概念上理解了 KV 缓存之后，接下来的大问题是它在实际小示例中的性能表现如何。为了测试实现效果，我们可以将前面提到的两个代码文件作为 Python 脚本运行，这些脚本会使用小型 124M 参数 LLM 生成 200 个新 token（以 4-token 提示 "Hello, I am" 开始）：


```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

在配备 M4 芯片（CPU）的 Mac Mini 上，结果如下：

|                        | Tokens/sec |
| ---------------------- | ---------- |
| `gpt_ch04.py`          | 27         |
| `gpt_with_kv_cache.py` | 144        |

因此，可以看到，对于一个小型 124M 参数模型和 200-token 的短序列，我们已经获得了约 5 倍的加速。（请注意，该实现是针对代码可读性优化的，并没有针对 CUDA 或 MPS 运行时速度进行优化，如果要优化，需要预先分配张量而不是每次重新生成并拼接。）

**注意：**模型在两种情况下生成的都是“乱码”，例如：

> 输出文本: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...

这是因为我们还没有训练模型。下一章会对模型进行训练，并且在训练后的模型上可以使用 KV 缓存（KV 缓存仅用于推理阶段）生成连贯文本。这里我们使用未训练的模型是为了保持代码简单。

更重要的是，`gpt_ch04.py` 和 `gpt_with_kv_cache.py` 的实现生成的文本完全相同。这说明 KV 缓存的实现是正确的——索引错误很容易导致结果不同。

&nbsp;

## KV 缓存的优缺点

随着序列长度增加，KV 缓存的好处和缺点如下：

- [好处] **计算效率提升**：如果不使用缓存，在步骤 *t* 的注意力计算中必须将新的 query 与前 *t* 个 key 进行比较，累积计算量呈二次增长 O(n²)。使用缓存后，每个 key 和 value 只计算一次，然后重复使用，使得每步的总复杂度降为线性 O(n)。

- [缺点] **内存使用线性增长**：每生成一个新 token 都会追加到 KV 缓存中。对于长序列和更大的 LLM，累积的 KV 缓存可能占用大量甚至过量的（GPU）内存。可以通过截断 KV 缓存来缓解，但这会增加复杂性（但在部署 LLM 时，这种权衡通常是值得的）。

&nbsp;

## KV 缓存实现优化

上面提供的 KV 缓存概念实现主要为了代码可读性和教学目的。在实际部署（尤其是较大模型和长序列）中，需要更仔细的优化。

&nbsp;
### 缓存扩展时的常见问题

- **内存碎片化与频繁分配**：如前所示连续使用 `torch.cat` 拼接张量，会因频繁分配内存而导致性能瓶颈。

- **内存使用线性增长**：如果没有适当处理，KV 缓存对于非常长的序列会变得不切实际。

&nbsp;
#### 提示 1：预分配内存

与其重复拼接张量，不如根据预期最大序列长度预先分配足够大的张量。这样可以保证内存使用一致并减少开销。伪代码示例如下：

```python
# Example pre-allocation for keys and values
max_seq_len = 1024  # maximum expected sequence length
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```
在推理过程中，我们可以直接写入这些预分配张量的切片。

&nbsp;
#### 提示 2：使用滑动窗口截断缓存

为了避免 GPU 内存过度增长，可以实现一个带动态截断的滑动窗口方法。通过滑动窗口，我们只保留缓存中最近的 `window_size` 个 token：



```python
# Sliding window cache implementation
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```
&nbsp;
#### 实践中的优化

你可以在 [`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py) 文件中找到这些优化实现。

在 Mac Mini M4 芯片（CPU）上，进行 200-token 的生成，并且窗口大小等于上下文长度（以保证结果一致）时，代码运行速度如下表所示：

|                                  | Tokens/sec |
| -------------------------------- | ---------- |
| `gpt_ch04.py`                    | 27         |
| `gpt_with_kv_cache.py`           | 144        |
| `gpt_with_kv_cache_optimized.py` | 166        |

不幸的是，在 CUDA 设备上，这些速度优势消失，因为这个模型非常小，设备间的数据传输和通信开销超过了 KV 缓存带来的好处。

&nbsp;
## 其他资源

1. [Qwen3 from-scratch KV 缓存基准](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation)
2. [Llama 3 from-scratch KV 缓存基准](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation)
3. [从零实现 LLM 的 KV 缓存原理与代码](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) —— 本 README 的更详细说明
