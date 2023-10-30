# NLP_Practice2023

随着ChatGPT和大模型席卷NLP领域，基于Transformer的生成式模型成为了每个NLP研究者必备的基础知识。本练习致力于从零构建一个基于Transformer的诗歌生成模型，从而熟悉Transformer的结构以及生成式模型的架构。

**数据集**：https://github.com/THUNLP-AIPoet/Datasets/tree/master/CCPC

**模型结构**：仿照GPT1的架构，不需要加载GPT预训练参数，Transformer层数、隐藏层大小等超惨可以根据你的硬件资源、实验效果等调整：

<img src="GPT.png" alt="drawing" width="300"/>

**格式**：

```
输入： 标题: <Title> 关键词: <keyword1> <keyword2> 诗歌: 
输出： <line1>|<line2>|<line3>|<line4> 
````

**要求**：
1. 尽可能基于更加底层的函数实现该模型，比如[torch.nn.MultiheadAttention](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)；
2. 可以先参考别人的代码，运行实现，但是最终提交的代码一定是自己手写的；
3. 仅使用训练集中的数据进行训练，禁止使用验证集/测试集训练；

**提交结果**：
1. 源代码：不需要打包数据集）；
2. 实验报告：README.md形式，包含以下内容：
    1. 超参数设置（层数、词表大小、学习率、训练轮数等等）；
    2. 在验证集和测试集上的perplexity；
    3. 测试集中前五首诗的生成结果；
    4. 实验过程中的发现和心得，或者其他任何你想要分享的内容；
