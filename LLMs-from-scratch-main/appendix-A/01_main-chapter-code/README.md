# 附录 A：PyTorch 入门

### 本章主要代码

- [code-part1.ipynb](code-part1.ipynb) 包含章节中 A.1 到 A.8 的所有代码  
- [code-part2.ipynb](code-part2.ipynb) 包含章节中 A.9 GPU 的所有代码  
- [DDP-script.py](DDP-script.py) 包含演示多 GPU 使用的脚本（注意 Jupyter Notebook 仅支持单 GPU，因此这是一个脚本，而非笔记本）。可运行方式：`python DDP-script.py`。如果你的机器有超过 2 个 GPU，可运行：`CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py`  
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章习题解答

### 可选代码

- [DDP-script-torchrun.py](DDP-script-torchrun.py) 是 `DDP-script.py` 的可选版本，通过 PyTorch 的 `torchrun` 命令运行，而不是通过 `multiprocessing.spawn` 自行生成和管理多个进程。使用 `torchrun` 命令的优点是可以自动处理分布式初始化，包括多节点协调，从而略微简化了设置过程。可通过 `torchrun --nproc_per_node=2 DDP-script-torchrun.py` 使用此脚本。
