import pandas as pd
import numpy as np
import datetime
import torch
import random
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import gpustat

# 设置随机种子 固定结果
def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置

    torch.backends.cudnn.deterministic = True  # 固定网络结构

def print_gpu_info():
    if torch.cuda.is_available():
        stats = gpustat.GPUStatCollection.new_query()
        for gpu in stats:
            print('GPU ID: {}   Used Memory: {}/{}'.format(gpu.index, gpu.memory_used, gpu.memory_total))
    else:
        print("没有显卡")

    
class Embedding(nn.Module):
    def __init__(self, block_size, batch_size, vocab_size, embedding_size):
        # 为了降维，embedding_size < vocab_size
        # GPT说在transformers中二者要相等，逻辑合理
        super().__init__()  # 添加这一行
        self.block_size = block_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

    def forward(self, idx):
        idx_one_hot = F.one_hot(idx, num_classes = self.vocab_size).float()
        # print('idx_one_hot.shape: ', idx_one_hot.shape)

        embedding = torch.randn((self.vocab_size, self.embedding_size), requires_grad=True)
        logits = idx_one_hot @ embedding
        # print('logits.shape: ', logits.shape)

        counts = logits.exp()
        probs = counts / counts.sum(-1, keepdims=True)
        # print('probs.shape: ', probs.shape)    
        # 上面就是softmax  probs = F.softmax(logits, dim=-1)

        return logits, probs # F.cross_entropy需要logits

if __name__=="__main__":

    set_seeds(42)

    block_size = 8
    batch_size = 4
    vocab_size = 10
    embedding_size = 20

    em = Embedding(block_size=block_size, batch_size=batch_size, vocab_size=vocab_size, embedding_size=embedding_size)

    idx = torch.randint(vocab_size, (batch_size, block_size))
    print(idx)

    print(em(idx))



# def Embedding(idx, block_size, batch_size, vocab_size, embedding_size):
#     idx_one_hot = F.one_hot(idx, num_classes = vocab_size).float()
#     # print('idx_one_hot.shape: ', idx_one_hot.shape)

#     embedding = torch.randn((vocab_size, embedding_size), requires_grad=True)
#     logits = idx_one_hot @ embedding
#     # print('logits.shape: ', logits.shape)

#     counts = logits.exp()
#     probs = counts / counts.sum(-1, keepdims=True)
#     # print('probs.shape: ', probs.shape)    
#     # 上面就是softmax  probs = F.softmax(logits, dim=-1)

#     return probs

# if __name__=="__main__":

#     block_size = 8
#     batch_size = 4
#     vocab_size = 10
#     embedding_size = 20

#     idx = torch.randint(vocab_size, (batch_size, block_size))
#     print(idx)

#     print(Embedding(idx, block_size, batch_size, vocab_size, embedding_size))






