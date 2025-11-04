# 构建与预训练 LLM 交互的用户界面

该附加文件夹包含运行类似 ChatGPT 的用户界面的代码，用于与第五章的预训练 LLM 交互，如下所示。

![Chainlit UI 示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-orig.webp)

为了实现该用户界面，我们使用开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 第 1 步：安装依赖

首先，通过以下命令安装 `chainlit` 包：


```bash
pip install chainlit
```

（或者，执行 `pip install -r requirements-extra.txt`。）

&nbsp;
## 第 2 步：运行 `app` 代码

该文件夹包含 2 个文件：

1. [`app_orig.py`](app_orig.py)：此文件加载并使用来自 OpenAI 的原始 GPT-2 权重。  
2. [`app_own.py`](app_own.py)：此文件加载并使用我们在第五章生成的 GPT-2 权重。这要求你先执行 [`../01_main-chapter-code/ch05.ipynb`](../01_main-chapter-code/ch05.ipynb) 文件。

（打开并检查这些文件以了解更多信息。）

从终端运行以下命令之一以启动 UI 服务器：


```bash
chainlit run app_orig.py
```

or

```bash
chainlit run app_own.py
```

运行上述命令之一后，应会打开一个新的浏览器标签页，在其中可以与模型交互。如果浏览器标签页未自动打开，请检查终端命令，并将本地地址复制到浏览器地址栏（通常地址为 `http://localhost:8000`）。
