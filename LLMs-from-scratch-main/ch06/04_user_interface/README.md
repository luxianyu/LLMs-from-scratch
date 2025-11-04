# 构建与基于 GPT 的垃圾邮件分类器交互的用户界面

这个附加文件夹包含了运行类似 ChatGPT 的用户界面代码，用于与第 6 章中微调的基于 GPT 的垃圾邮件分类器进行交互，如下图所示。

![Chainlit UI 示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-spam.webp)

要实现这个用户界面，我们使用开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 步骤 1：安装依赖

首先，通过以下命令安装 `chainlit` 包：


```bash
pip install chainlit
```

（或者，可以执行 `pip install -r requirements-extra.txt`。）

&nbsp;
## 步骤 2：运行 `app` 代码

[`app.py`](app.py) 文件包含了 UI 代码。打开并查看这些文件以了解更多信息。

该文件会加载并使用我们在第 6 章生成的 GPT-2 分类器权重。因此，需要先执行 [`../01_main-chapter-code/ch06.ipynb`](../01_main-chapter-code/ch06.ipynb) 文件。

从终端执行以下命令以启动 UI 服务器：

```bash
chainlit run app.py
```

运行上述命令后，应该会在浏览器中打开一个新标签页，您可以与模型进行交互。如果浏览器标签页没有自动打开，请检查终端命令，并将本地地址复制到浏览器地址栏（通常地址为 `http://localhost:8000`）。
