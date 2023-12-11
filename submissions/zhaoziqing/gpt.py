import random

import torch
import numpy as np
from torch import nn
import config
from mydataset import MySequence

ns = MySequence()
word2id = ns.dict
id2word = ns.inverse_dict
device = config.device
unk_id = 1
pad_id = 0
vocab_size = len(word2id)+1
seq_len = 512
n_layers = 6
d_model = 768
n_heads = 8
d_k = d_v = 64
d_ff = 2048
drop = 0.1


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(drop)

    def forward(self, q, k, v, mask):
        """
        :param q: (bs, n_heads, q_len, d_k)
        :param k: (bs, n_heads, k_len, d_k)
        :param v: (bs, n_heads, v_len, d_v)
        :param mask: (bs, n_heads, q_len, k_len)
        :return:
        """
        scores = torch.matmul(q, k.transpose(-1, -2)) / (d_k ** 0.5)
        scores.masked_fill_(mask, -1e9)     # mask
        # softmax需要指定维度  (bs, n_heads, q_len, k_len)
        weights = nn.Softmax(dim=-1)(scores)
        weights = self.dropout(weights)
        # k_len=v_len (bs, n_heads, q_len, d_v)
        output = torch.matmul(weights, v)
        return output, weights


class MultiAttention(nn.Module):
    def __init__(self):
        super(MultiAttention, self).__init__()
        self.Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.attention = Attention()
        self.linear = nn.Linear(d_v * n_heads, d_model, bias=False)

    def forward(self, q, k, v, mask):
        """
        :param q:  (bs, q_len, d_model)
        :param k:  (bs, k_len, d_model)
        :param v:  (bs, v_len, d_model)
        :param mask: (bs, q_len, k_len)
        :return:
        """
        bs = q.size(0)
        # （bs, n_heads, q_len, d_k）
        q_heads = self.Q(q).view(bs, -1, n_heads, d_k).transpose(1, 2)
        # （bs, n_heads, k_len, d_k）
        k_heads = self.K(k).view(bs, -1, n_heads, d_k).transpose(1, 2)
        # （bs, n_heads, v_len, d_v）
        v_heads = self.V(v).view(bs, -1, n_heads, d_v).transpose(1, 2)
        # (bs, n_heads, q_len, k_len)
        mask = mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # (bs, n_heads, q_len, d_v),(bs, n_heads, q_len, k_len)
        attn, attn_weights = self.attention(q_heads, k_heads, v_heads, mask)
        attn = attn.transpose(1, 2).contiguous().view(
            bs, -1, n_heads*d_v)  # (bs, q_len, n_heads*d_v)
        output = self.linear(attn)  # (bs, q_len, d_model)
        return output, attn_weights


class Position(nn.Module):
    def __init__(self):
        super(Position, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=False)
        self.linear2 = nn.Linear(d_ff, d_model, bias=False)
        self.relu = nn.ReLU()
        # nn.init.normal_(self.linear1.weight, std=0.02)
        # nn.init.normal_(self.linear2.weight, std=0.02)

    def forward(self, x):
        y = self.relu(self.linear1(x))
        y = self.linear2(y)
        return y


class Decoderlayer(nn.Module):
    def __init__(self):
        super(Decoderlayer, self).__init__()
        self.ma = MultiAttention()
        self.drop1 = nn.Dropout(drop)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.position = Position()
        self.drop2 = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x, mask):
        """
        :param x: (bs, seq_len, d_model)
        :param mask: (bs, seq_len, seq_len)
        :return:
        """
        attn, attn_weights = self.ma(
            x, x, x, mask)     # (bs, seq_len, d_model), (bs, n_heads, seq_len, seq_len)
        attn = self.drop1(attn)
        attn = self.norm1(attn + x)
        ffn = self.position(attn)   # (bs, seq_len, d_model)
        ffn = self.drop2(ffn)
        ffn = self.norm2(ffn + attn)
        return ffn, attn_weights


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len+1, d_model)
        self.dropout = nn.Dropout(drop)
        self.layers = nn.ModuleList([Decoderlayer() for _ in range(n_layers)])
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        """
        :param x: (bs, seq_len)
        :return: (bs, seq_len, d_model), (n_layers, bs, n_heads, seq_len, seq_len )
        """
        pos = torch.arange(x.size(1), device=device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # (bs, seq_len)
        # pos_pad_mask = x.eq(self.pad_id)
        # pos.masked_fill_(pos_pad_mask, 0)
        y = self.dropout(self.emb(x) + self.pos_emb(pos))

        pad_mask = self.get_pad_mask(x, x)
        seq_mask = self.get_seq_mask(x).to(device=device)
        mask = torch.gt((pad_mask + seq_mask), 0)  # 是否>1
        weights = []
        for layer in self.layers:
            y, y_weight = layer(y, mask)
            weights.append(y_weight)

        return y, weights

    def get_pad_mask(self, q, k):
        """
        :param q: (bs, q_len)
        :param k: (bs, k_len)
        :return: (bs, q_len, k_len)
        q_len = k_len
        """
        pad_mask = k.data.eq(pad_id).unsqueeze(1).repeat(1, q.size(1), 1)
        return pad_mask

    def get_seq_mask(self, q):
        """
        :param q: (bs, q_len)
        :return: (bs, q_len, q_len)
        """
        bs, q_len = q.size()
        seq_mask = torch.ones(bs, q_len, q_len).triu(diagonal=1)  # 上三角矩阵
        return seq_mask


class GPT(nn.Module):
    def __init__(self):
        super(GPT, self).__init__()
        self.decoder = Decoder()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        y, weight = self.decoder(x)  # (bs, seq_len, d_model)
        y = self.linear(y)  # (bs, seq_len, vocab_size)
        return y.view(-1, y.size(-1)), weight

    def greedy(self, x):
        """
        :param x: (1, seq_len)
        :return:
        """
        start_len = len(x[0])
        end = False
        while not end:
            if len(x[0]) - start_len > 100:
                next = word2id.get("<SEP>")
                x = torch.cat([x.detach(), torch.tensor(
                    [[next]], device=device, dtype=x.dtype)], -1)
                break
            y, _ = self.decoder(x)
            y = self.linear(y)
            prob = y.squeeze(0).max(dim=-1, keepdim=False)[1]  # 取最大值
            next = prob.data[-1]
            if next == word2id.get("<SEP>"):
                end = True
            x = torch.cat([x.detach(), torch.tensor(
                [[next]], device=device, dtype=x.dtype)], -1)
        return x.squeeze(0)
    
    def answer(self, sentence):
        sentence = [word2id.get(i, unk_id) for i in sentence]
        sentence = torch.LongTensor(sentence).to(device).unsqueeze(0)
        output = self.greedy(sentence)
        output = [id2word[int(i)] for i in output]
        sep_idx =[]
        for i in range(len(output)):
            if output[i] == "<SEP>":
                sep_idx.append(i)
        answer = output[sep_idx[-2]+1: sep_idx[-1]]
        answer = "".join(answer)
        return answer