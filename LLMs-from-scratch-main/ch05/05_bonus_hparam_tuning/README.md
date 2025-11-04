# 预训练超参数优化

[hparam_search.py](hparam_search.py) 脚本基于 [附录 D: 为训练循环增加附加功能](../../appendix-D/01_main-chapter-code/appendix-D.ipynb) 中的扩展训练函数，旨在通过网格搜索（grid search）寻找最优超参数。

>[!注意]
该脚本运行时间较长。你可能希望减少顶部 `HPARAM_GRID` 字典中探索的超参数配置数量。
