# 加载预训练权重的替代方法

该文件夹包含在 OpenAI 权重不可用时的替代加载策略。

- [weight-loading-pytorch.ipynb](weight-loading-pytorch.ipynb)：（推荐）包含从 PyTorch 状态字典加载权重的代码，这些状态字典是我通过将原始 TensorFlow 权重转换而创建的

- [weight-loading-hf-transformers.ipynb](weight-loading-hf-transformers.ipynb)：包含通过 `transformers` 库从 Hugging Face 模型库加载权重的代码

- [weight-loading-hf-safetensors.ipynb](weight-loading-hf-safetensors.ipynb)：包含通过 `safetensors` 库直接从 Hugging Face 模型库加载权重的代码（跳过 Hugging Face Transformer 模型的实例化）
