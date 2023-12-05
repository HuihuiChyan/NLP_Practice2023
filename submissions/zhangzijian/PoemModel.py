import torch
from torch import nn
import pandas as pd
import math
from torch.utils.data import DataLoader,Dataset
import time    
import pdb

class Embedding_layer(nn.Module):
    # 词嵌入层
    def __init__(self,vocab_size,embed_dim,device):
        super(Embedding_layer,self).__init__()
        self.embed_dim = embed_dim # 词嵌入维度
        self.vocab_size = vocab_size # 词表大小
        self.device = device # 并行设备
        self.embedding = torch.nn.Embedding(vocab_size,embed_dim)
        # self.PositionEncoding = torch.nn.Embedding(max_pos,embed_dim) # 位置编码
        
    def forward(self,x):
        # len_s = x.size(1)
        # pos = torch.arange(len_s,dtype=torch.long,device=self.device)
        # pos = pos.unsqueeze(0).expand_as(x)
        x = self.embedding(x) # batch_size * max_seq_len * d_model
        x += self.PositionEncoding(x) 
        return x
   
    def PositionEncoding(self,x):
        # 位置编码函数
        batch_size = x.size(0)
        rows = x.size(1)
        dim = self.embed_dim
        PE = torch.zeros([batch_size,rows,dim],dtype=torch.float32).to(self.device)
        
        pos_mat = torch.arange(rows).reshape((-1,1))
        exponent = torch.arange(0,dim,2).reshape((1,-1)) / dim
        X = pos_mat / torch.pow(10000,exponent)
        PE[:,:,0::2] = torch.sin(X)
        PE[:,:,1::2] = torch.cos(X)
        return PE

class Compute_Attention(nn.Module):
    # 计算注意力得分 点积计算注意力得分
    def __init__(self,device):
        super(Compute_Attention,self).__init__()
        self.device = device
        self.sf = nn.Softmax(dim = -1) # softmax层
        
    def forward(self,Q,K,V):
        score = torch.matmul(Q,K.transpose(2,-1)) / math.sqrt(K.size(-1))
        # pdb.set_trace() 
        subs_mask = self.get_subsequence_mask(score.size(0),score.size(1),score.size(2))
        score.masked_fill_(subs_mask,-1e7) # 对score进行mask
        score = self.sf(score) # (batch_size,num_heads,len_s,len_s)
        # pdb.set_trace()
        score = torch.matmul(score,V) # (batch_size,num_heads,len_s,d_v)
        return score
    
    def get_subsequence_mask(self,dim_1,dim_2,dim_3):
        # sequence_mask，是前t个token预测第t+1个token
        mask = torch.triu(torch.ones([dim_1,dim_2,dim_3,dim_3],dtype=torch.bool),diagonal = 1).to(self.device)
        return mask
    
class MultiHeadAttention(nn.Module):
    # MultiHeadAttention
    def __init__(self,num_heads,d_model,batch_size,device):
        super(MultiHeadAttention,self).__init__()
        self.d_k = self.d_v = self.d_q = 64
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.d_model = d_model
        self.device = device
        self.W_Q = nn.Linear(d_model,self.d_q * num_heads,bias = False) # Q向量权重矩阵
        self.W_K = nn.Linear(d_model,self.d_k * num_heads,bias = False) # K向量权重矩阵
        self.W_V = nn.Linear(d_model,self.d_v * num_heads,bias = False) # V向量权重矩阵
        self.W_O = nn.Linear(num_heads * self.d_v,d_model,bias = False) # 多头注意力输出权重矩阵
        self.layernorm = nn.LayerNorm(d_model) # 正则化
    
    def forward(self,x):
        residual = x # 残差 ()
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x) # batch_size * len_s * (num_heads * d_x)(d_x 指的是q、k、v所对应的维度)
        
        Q = torch.reshape(Q,(self.batch_size,-1,self.num_heads,self.d_q)).transpose(1,2) # (batch_size,num_heads,len_s,d_q)
        K = torch.reshape(K,(self.batch_size,-1,self.num_heads,self.d_k)).transpose(1,2) # (batch_size,num_heads,len_s,d_k)
        V = torch.reshape(V,(self.batch_size,-1,self.num_heads,self.d_v)).transpose(1,2) # (batch_size,num_heads,len_s,d_v)
         
        # 注意力得分计算
        attn_computer = Compute_Attention(self.device)
        attn = attn_computer(Q,K,V).transpose(1,2) # (batch_size,len_s,num_heads,d_v)
        attn = torch.reshape(attn,(self.batch_size,-1,self.d_v * self.num_heads)) # (batch_size,len_s,d_v * num_heads)
        attn = self.W_O(attn) # (batch_size,len_s,d_model)
        attn += residual # 残差连接
        attn = self.layernorm(attn) # 正则化
        return attn

class FeedForwardNet(nn.Module):
    # 前馈神经网络层
    def __init__(self,d_model):
        super(FeedForwardNet,self).__init__()
        self.d_ffn = 2048 # 隐藏层维度
        self.d_model = d_model
        self.ff = nn.Sequential(nn.Linear(d_model,self.d_ffn,bias = False),
                                nn.ReLU(),
                                nn.Linear(self.d_ffn,d_model,bias = False)) # 全连接三层神经网络
        self.layernorm = nn.LayerNorm(d_model)
    def forward(self,x):
        residual = x
        x = self.ff(x)
        x = self.layernorm(x + residual)
        return x

class Decoder_layer(nn.Module):
    # 一层decoder 由多头注意力和前馈神经网络组成
    def __init__(self,num_heads,d_model,batch_size,device):
        super(Decoder_layer,self).__init__()
        self.multiheads = MultiHeadAttention(num_heads,d_model,batch_size,device) # 多头注意力
        self.ffn = FeedForwardNet(d_model) # 前馈神经网络
    
    def forward(self,x):
        x = self.multiheads.forward(x)
        x = self.ffn.forward(x)
        return x

class Decoder(nn.Module):
    # 多层Transformer Decoder_layer
    def __init__(self,num_heads,d_model,num_layers,batch_size,device):
        super(Decoder,self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList([Decoder_layer(num_heads,d_model,batch_size,device) for _ in range(num_layers)]) # 多层decoder
    
    def forward(self,x):
        for layer in self.decoder_layers:
            x = layer.forward(x)
        return x
                      
class PoemModel(nn.Module):
    def __init__(self, vocab_size,d_model,num_heads,num_layers,device,batch_size):
        super(PoemModel,self).__init__()
        self.d_model = d_model # 模型的维度
        self.embedding = Embedding_layer(vocab_size,d_model,device) # 嵌入层
        self.decoder = Decoder(num_heads,d_model,num_layers,batch_size,device) # 解码器 batch_size * len_s * d_model
        self.linear = nn.Linear(d_model,vocab_size) # 线性层 batch_size * len_s * vocab_size
        #self.sf = nn.Softmax(dim = 2) 
    
    def forward(self,x):
        x = self.embedding.forward(x) # 嵌入层
        x = self.decoder.forward(x) # 解码器
        x = self.linear(x) # 线性层
        #x = self.sf(x)
        return x
    
    