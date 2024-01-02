import json

import numpy as np
import torch
from config import Config
from torch import nn

config = Config()
device = config.device
with open(config.dict_url, 'r', encoding='utf-8') as f:
    dicts = json.load(f)
f.close()
char2index, index2char = dicts['char2index'], dicts['index2char']
index2char.append("\n")
char2index["\n"] = len(index2char) - 1
vocab_size = len(index2char)
print(f"vocab_size = {vocab_size}")

embedding_size = config.embedding_size
ffn_size = config.ffn_size
k_size = v_size = config.k_size
num_layers = config.num_layers
num_heads = config.num_heads
dropout = config.dropout
max_len = config.max_len


class PoswiseFeedForward(nn.Module):
    """
        前馈神经网络 类似于MLP
    """

    def __init__(self):
        super(PoswiseFeedForward, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, ffn_size, bias=False),
            nn.ReLU(),
            # SwiGLU(),
            nn.Linear(ffn_size, embedding_size, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs


def get_attn_mask(tensor_q, tensor_k):
    """
    :param tensor_q: query
    :param tensor_k: key
    :return: 注意力掩码

    默认<pad>映射为0
    tensor.unsqueeze(x)给张量的第x维增加维度,值为1 例如size(4,3,2)调unsqueeze(1)得到size(4,1,3,2)
    tensor.data 复制原tensor 变更会同时更改二者
    np.triu()返回上三角部分，k=1
    """
    batch_size, sql_len_q = tensor_q.size()
    sql_len_k = tensor_k.size(1)
    pad_mask = tensor_k.data.eq(0).unsqueeze(1).byte()
    pad_mask = pad_mask.expand(batch_size, sql_len_q, sql_len_k)

    seq_mask = np.ones([batch_size, sql_len_q, sql_len_q])
    seq_mask = np.triu(seq_mask, k=1)  # 获得上三角矩阵 其中主对角线及其左下均为0
    seq_mask = torch.from_numpy(seq_mask).byte()  # MultiheadAttention的attn_mask当类型为byte时使用True/False

    pad_mask = pad_mask.to(device)
    seq_mask = seq_mask.to(device)
    """
        torch.gt(t1,t2) t1的对应元素大于t2的时返回true 否则false
        mask--pad的部分在pad_mask为1，应该被遮蔽的上三角在subsequence_mask的部分为1，大于0的应该被遮蔽 为True
        在MultiheadAttention的解释中 attn_mask为byte类型的True时，表示应当忽略它
    """
    return torch.gt((pad_mask + seq_mask), 0)


def ScaledDotProductAttention(Q, K, V, attn_mask):
    """
        点积注意力得分计算

        首先计算注意力得分里softmax部分，Q·K(T)/(根号)(k_size)
        再根据attn_mask将不应该看到的地方遮蔽(就地更改;-1e9经过softmax之后趋近于0)
        随后计算softmax，得到注意力分数attn
        最后点积V得到输出内容
        torch.sqrt()的输入要是tensor 而np.sqrt()没有该限制
    """
    scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(k_size)
    scores.masked_fill_(attn_mask, -1e9)
    attn = nn.Softmax(dim=-1)(scores)
    pred = torch.matmul(attn, V)
    return pred, attn


class SwiGLU(nn.Module):
    def __init__(self):
        super(SwiGLU, self).__init__()
        self.linear1 = nn.Linear(ffn_size, ffn_size, bias=False)
        self.linear2 = nn.Linear(ffn_size, ffn_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, X):
        return torch.mul(self.silu(self.linear1(X)), self.linear2(X))


class MultiheadAttention(nn.Module):
    def __init__(self):
        super(MultiheadAttention, self).__init__()
        self.W_Q = nn.Linear(embedding_size, k_size * num_heads, bias=False)
        self.W_K = nn.Linear(embedding_size, k_size * num_heads, bias=False)
        self.W_V = nn.Linear(embedding_size, v_size * num_heads, bias=False)
        self.fc = nn.Linear(v_size * num_heads, embedding_size, bias=False)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view(batch_size, -1, num_heads, k_size).transpose(1, 2)
        # Q [batch_size,sql_len,embedding_size)→[batch_size,sql_len,k_size*num_heads]
        # →[batch_size,sql_len,num_heads,k_size]→[batch_size,num_heads,sql_len,k_size]
        K = self.W_K(K).view(batch_size, -1, num_heads, k_size).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, num_heads, v_size).transpose(1, 2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)  # 多头注意力 需要将其提升一个维度
        # attn_mask [batch_size,sql_len,sql_len]→[batch_size,num_heads,sql_len,sql_len]
        pred, attn = ScaledDotProductAttention(Q=Q, K=K, V=V, attn_mask=attn_mask)
        pred = pred.transpose(1, 2).reshape(batch_size, -1, num_heads * v_size)  # 在注意力得分计算时最后点积V 因此是num_heads*v_size
        # pred [batch_size,num_heads,sql_len,v_size]→[batch_size,sql_len,num_heads,v_size]
        # →[batch_size,sql_len,num_heads*v_size]
        pred = self.fc(pred)
        return pred, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.attn_layer = MultiheadAttention()
        self.layernorm1 = nn.LayerNorm(embedding_size)
        self.ffn = PoswiseFeedForward()
        self.layernorm2 = nn.LayerNorm(embedding_size)

    def forward(self, input, attn_mask):
        old_input = input
        output, attn = self.attn_layer(input, input, input, attn_mask)
        output = self.layernorm1(output + old_input)
        old_input = output
        output = self.ffn(output)
        output = self.layernorm2(output + old_input)
        return output, attn


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, num_hiddens, dropout, max_len=max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2,
                                                                                                      dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(device)
        return self.dropout(X)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.source_embedding = nn.Embedding(vocab_size, embedding_size)
        self.pos_embedding = nn.Embedding(max_len, embedding_size)
        # self.pos_encoder = PositionalEncoding(num_hiddens=embedding_size, dropout=0.1)
        self.decoder_layers = nn.ModuleList([DecoderLayer() for _ in range(num_layers)])

    def forward(self, inputs):
        batch_size, sql_len = inputs.size()
        pos = torch.arange(sql_len, dtype=torch.long, device=device)
        pos = pos.unsqueeze(0).expand(batch_size, sql_len)
        # decoder_input = self.source_embedding(inputs) + self.pos_encoder(self.pos_embedding(pos))
        decoder_input = self.source_embedding(inputs) + self.pos_embedding(pos)


        attn_mask = get_attn_mask(inputs, inputs)
        output = decoder_input
        attns = []

        for decoder_layer in self.decoder_layers:
            decoder_output, attn = decoder_layer(decoder_input, attn_mask)
            decoder_input = decoder_output
            output = decoder_output
            attns.append(attn)

        return output, attns


class GPT(nn.Module):
    def __init__(self):
        super(GPT, self).__init__()
        self.decoder = Decoder()
        self.linear = nn.Linear(embedding_size, vocab_size)  # 将嵌入转化回词表

    def forward(self, inputs):
        output, attns = self.decoder(inputs)
        possible = self.linear(output)
        possible = possible.view(-1, possible.size(-1))
        # possible [batch_size,sql_len,embedding_size]→[batch_size,sql_len,vocab_size]
        # →[batch_size*sql_len,vocab_size]
        return possible, attns

    def greedy_decoder(self, input):
        """
        贪婪搜索 直至预测的终止符才停止 <sep>--2
        :param input: 文本编码后的输入
        :return: 生成的文本（需要解码）

        tensor.detach() 阻止反向梯度传播
        tensor.squeeze() 删除值为1的部分
        torch.cat([x,y],dim=-1)代表将x和y直接拼接
        tensor.max(dim,keepdim)[x] x为0是最大值的索引 x为1是最大值
        """
        while True:
            output, attns = self.decoder(input)
            possibilty = self.linear(output)
            pred_word = possibilty.squeeze().max(dim=-1, keepdim=False)[1].data[-1]
            input = torch.cat([input, torch.tensor([[pred_word]], dtype=input.dtype, device=device)], dim=-1)
            if pred_word == char2index['<sep>']:
                break
        return input

    def answer(self, sentence):
        input = [char2index[word] for word in sentence]
        input = torch.tensor(input, dtype=torch.long, device=device).unsqueeze(0)

        output = self.greedy_decoder(input).squeeze(0)
        output_index = [index2char[int(index)] for index in output]
        # 最后两个<sep>之间的内容是生成结果（对应诗句）
        sep_indexs = []
        for i in range(len(output_index)):
            if output_index[i] == "<sep>":
                sep_indexs.append(i)
        answer = output_index[sep_indexs[-2] + 1:-1]
        answer = "".join(answer)
        return answer