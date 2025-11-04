# 第5章：在无标签数据上进行预训练

### 本章主要代码

- [ch05.ipynb](ch05.ipynb) 包含了本章中出现的所有代码  
- [previous_chapters.py](previous_chapters.py) 是一个 Python 模块，包含前几章的 `MultiHeadAttention` 模块和 `GPTModel` 类，我们在 [ch05.ipynb](ch05.ipynb) 中导入该模块来对 GPT 模型进行预训练  
- [gpt_download.py](gpt_download.py) 包含用于下载 GPT 预训练模型权重的实用函数  
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章的练习答案

### 可选代码

- [gpt_train.py](gpt_train.py) 是一个独立的 Python 脚本文件，包含我们在 [ch05.ipynb](ch05.ipynb) 中实现的训练 GPT 模型的代码（可视为本章代码的总结版）  
- [gpt_generate.py](gpt_generate.py) 是一个独立的 Python 脚本文件，包含我们在 [ch05.ipynb](ch05.ipynb) 中实现的加载并使用 OpenAI 预训练权重的代码
