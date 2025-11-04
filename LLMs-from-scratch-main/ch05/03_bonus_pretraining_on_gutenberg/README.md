# 在 Project Gutenberg 数据集上预训练 GPT

该目录中的代码包含在 Project Gutenberg 提供的免费书籍上训练小型 GPT 模型的代码。

正如 Project Gutenberg 网站所述，“绝大多数 Project Gutenberg 电子书在美国属于公有领域。”

有关使用 Project Gutenberg 提供的资源的更多信息，请阅读 [Project Gutenberg Permissions, Licensing and other Common Requests](https://www.gutenberg.org/policy/permission.html) 页面。

&nbsp;
## 如何使用此代码

&nbsp;

### 1）下载数据集

在本节中，我们使用 [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) GitHub 仓库中的代码从 Project Gutenberg 下载书籍。

截至本文撰写时，这将需要大约 50 GB 的磁盘空间，并且大约需要 10-15 小时，但实际时间可能更长，具体取决于 Project Gutenberg 自那时起增长的数据量。

&nbsp;
#### Linux 和 macOS 用户的下载说明

Linux 和 macOS 用户可以按照以下步骤下载数据集（如果你是 Windows 用户，请参见下面的说明）：

1. 将 `03_bonus_pretraining_on_gutenberg` 文件夹设置为工作目录，以便在该文件夹中本地克隆 `gutenberg` 仓库（运行提供的 `prepare_dataset.py` 和 `pretraining_simple.py` 脚本时必需）。例如，在 `LLMs-from-scratch` 仓库的文件夹中，可以通过以下方式进入 *03_bonus_pretraining_on_gutenberg* 文件夹：

```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

2. Clone the `gutenberg` repository in there:
```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. Navigate into the locally cloned `gutenberg` repository's folder:
```bash
cd gutenberg
```

4. Install the required packages defined in *requirements.txt* from the `gutenberg` repository's folder:
```bash
pip install -r requirements.txt
```

5. Download the data:
```bash
python get_data.py
```

6. Go back into the `03_bonus_pretraining_on_gutenberg` folder
```bash
cd ..
```

&nbsp;
#### Windows 用户的特殊说明

[`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) 代码兼容 Linux 和 macOS。然而，Windows 用户需要进行一些小调整，例如在 `subprocess` 调用中添加 `shell=True` 并替换 `rsync`。

或者，在 Windows 上运行此代码的更简单方法是使用 “Windows Subsystem for Linux”（WSL）功能，该功能允许用户在 Windows 中使用 Ubuntu 运行 Linux 环境。有关更多信息，请阅读 [Microsoft 官方安装说明](https://learn.microsoft.com/en-us/windows/wsl/install) 和 [教程](https://learn.microsoft.com/en-us/training/modules/wsl-introduction/)。

在使用 WSL 时，请确保已安装 Python 3（通过 `python3 --version` 检查，或者例如使用 `sudo apt-get install -y python3.10` 安装 Python 3.10），并在其中安装以下软件包：


```bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt-get install -y python3-pip && \
sudo apt-get install -y python-is-python3 && \
sudo apt-get install -y rsync
```

> **注意：**
> 关于如何设置 Python 及安装软件包的说明，请参见 [Optional Python Setup Preferences](../../setup/01_optional-python-setup-preferences/README.md) 和 [Installing Python Libraries](../../setup/02_installing-python-libraries/README.md)。
>
> 可选地，本仓库提供了一个运行 Ubuntu 的 Docker 镜像。关于如何使用提供的 Docker 镜像运行容器的说明，请参见 [Optional Docker Environment](../../setup/03_optional-docker-environment/README.md)。

&nbsp;
### 2）准备数据集

接下来，运行 `prepare_dataset.py` 脚本，该脚本将（截至本文撰写时的 60,173 个）文本文件合并成更少的、更大的文件，以便更高效地传输和访问：

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

```
...
Skipping gutenberg/data/raw/PG29836_raw.txt as it does not contain primarily English text.                                     Skipping gutenberg/data/raw/PG16527_raw.txt as it does not contain primarily English text.                                     100%|██████████████████████████████████████████████████████████| 57250/57250 [25:04<00:00, 38.05it/s]
42 file(s) saved in /Users/sebastian/Developer/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed
```


> **提示：**
> 请注意，生成的文件以纯文本格式存储，并未进行预分词处理以简化操作。然而，如果你计划频繁使用该数据集或进行多轮 epoch 训练，你可能希望更新代码以将数据集以预分词形式存储，从而节省计算时间。更多信息请参见本页底部的 *Design Decisions and Improvements*。

> **提示：**
> 你可以选择更小的文件大小，例如 50 MB。这会产生更多文件，但对于小规模文件的快速预训练测试可能会很有用。

&nbsp;
### 3）运行预训练脚本

可以按如下方式运行预训练脚本。请注意，附加的命令行参数示例使用了默认值以便说明：

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

输出将以以下格式显示：


> Total files: 3
> Tokenizing file 1 of 3: data_small/combined_1.txt
> Training ...
> Ep 1 (Step 0): Train loss 9.694, Val loss 9.724
> Ep 1 (Step 100): Train loss 6.672, Val loss 6.683
> Ep 1 (Step 200): Train loss 6.543, Val loss 6.434
> Ep 1 (Step 300): Train loss 5.772, Val loss 6.313
> Ep 1 (Step 400): Train loss 5.547, Val loss 6.249
> Ep 1 (Step 500): Train loss 6.182, Val loss 6.155
> Ep 1 (Step 600): Train loss 5.742, Val loss 6.122
> Ep 1 (Step 700): Train loss 6.309, Val loss 5.984
> Ep 1 (Step 800): Train loss 5.435, Val loss 5.975
> Ep 1 (Step 900): Train loss 5.582, Val loss 5.935
> ...
> Ep 1 (Step 31900): Train loss 3.664, Val loss 3.946
> Ep 1 (Step 32000): Train loss 3.493, Val loss 3.939
> Ep 1 (Step 32100): Train loss 3.940, Val loss 3.961
> Saved model_checkpoints/model_pg_32188.pth
> Book processed 3h 46m 55s
> Total time elapsed 3h 46m 55s
> ETA for remaining books: 7h 33m 50s
> Tokenizing file 2 of 3: data_small/combined_2.txt
> Training ...
> Ep 1 (Step 32200): Train loss 2.982, Val loss 4.094
> Ep 1 (Step 32300): Train loss 3.920, Val loss 4.097
> ...


&nbsp;
> **提示：**
> 实际操作中，如果你使用 macOS 或 Linux，我建议使用 `tee` 命令，将日志输出保存到 `log.txt` 文件，同时在终端打印：


```bash
python -u pretraining_simple.py | tee log.txt
```

&nbsp;
> **警告：**
> 请注意，在 `gutenberg_preprocessed` 文件夹中的 1 个约 500 MB 的文本文件上训练，大约需要 4 小时（使用 V100 GPU）。  
> 该文件夹包含 47 个文件，完成训练大约需要 200 小时（超过 1 周）。你可能希望在较少数量的文件上运行。

&nbsp;
## 设计决策与改进

请注意，此代码重点是保持简单和最小化，以用于教学目的。可以通过以下方式改进代码，以提高建模性能和训练效率：

1. 修改 `prepare_dataset.py` 脚本，从每本书的文件中去除 Gutenberg 固定模板文本。
2. 更新数据准备和加载工具，将数据集预先分词并以分词形式保存，这样每次调用预训练脚本时就不必重新分词。
3. 更新 `train_model_simple` 脚本，添加 [附录 D: 为训练循环增加附加功能](../../appendix-D/01_main-chapter-code/appendix-D.ipynb) 中介绍的功能，即余弦衰减（cosine decay）、线性预热（linear warmup）和梯度裁剪（gradient clipping）。
4. 更新预训练脚本以保存优化器状态（参见第五章 *5.4 在 PyTorch 中加载和保存权重*；[ch05.ipynb](../../ch05/01_main-chapter-code/ch05.ipynb)），并添加选项以加载现有模型和优化器检查点，如果训练中断可以继续训练。
5. 添加更高级的日志记录器（例如 Weights and Biases），以实时查看损失和验证曲线。
6. 添加分布式数据并行（DDP），在多 GPU 上训练模型（参见附录 A *A.9.3 多 GPU 训练*；[DDP-script.py](../../appendix-A/01_main-chapter-code/DDP-script.py)）。
7. 将 `previous_chapter.py` 脚本中从零实现的 `MultiheadAttention` 类替换为在 [Efficient Multi-Head Attention Implementations](../../ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb) 附加章节中实现的高效 `MHAPyTorchScaledDotProduct` 类，该类通过 PyTorch 的 `nn.functional.scaled_dot_product_attention` 函数使用 Flash Attention。
8. 通过 [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) (`model = torch.compile`) 或 [thunder](https://github.com/Lightning-AI/lightning-thunder) (`model = thunder.jit(model)`) 优化模型以加速训练。
9. 实现梯度低秩投影（Gradient Low-Rank Projection, GaLore）以进一步加快预训练过程。这可以通过将 `AdamW` 优化器替换为 [GaLore Python 库](https://github.com/jiaweizzhao/GaLore) 提供的 `GaLoreAdamW` 实现。

