import torch
import torch.nn as nn
import torch.nn.functional as F

# GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedding_layer(nn.Module):
    # 词嵌入层
    def __init__(self, vocab_size, embed_dim, device):
        super(Embedding_layer, self).__init__()
        self.embed_dim = embed_dim  # 词嵌入维度
        self.vocab_size = vocab_size  # 词表大小
        self.device = device  # GPU 设备
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        x = self.embedding(x)
        # print(x.shape)  # batch_size * input_len * d_model
        # print(self.PositionEncoding(x).shape)  # batch_size * input_len * d_model
        x += self.PositionEncoding(x)
        return x

    def PositionEncoding(self, x):
        # 位置编码函数
        batch_size = x.size(0)
        rows = x.size(1)
        dim = self.embed_dim
        PE = torch.zeros([batch_size, rows, dim], dtype=torch.float32).to(self.device)

        pos_mat = torch.arange(rows).reshape((-1, 1))
        exponent = torch.arange(0, dim, 2).reshape((1, -1)) / dim
        X = pos_mat / torch.pow(10000, exponent)
        PE[:, :, 0::2] = torch.sin(X)
        PE[:, :, 1::2] = torch.cos(X)
        return PE

class MultiheadAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # (batch_size, num_heads, seq_len, embed_dim / self.num_heads)
        q = self.q_linear(query).view(batch_size, seq_len, self.num_heads, self.embed_dim // self.num_heads).transpose(
            1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.embed_dim ** 0.5)

        if attn_mask is not None:
            attn_scores += attn_mask

        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2) == 0,
                                                  float('-1e7'))  # 不能设为 -inf，否则 loss = nan

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = torch.nn.Dropout(0.1)(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        output = self.out_linear(attn_output)

        return output


class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedforward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class GPTBlock(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super(GPTBlock, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x, key_padding_mask, attn_mask):
        attn_output = self.self_attn(x, x, x, key_padding_mask, attn_mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)
        ffn = self.feedforward(x)
        ffn = self.dropout2(ffn)
        x = self.norm2(x + ffn)

        return x


class GPTGenerator(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff, dropout, device):
        super(GPTGenerator, self).__init__()
        self.d_model = d_model  # 模型的维度

        self.embedding = Embedding_layer(vocab_size, d_model, device)  # 嵌入层
        self.GPT_blocks = nn.ModuleList([GPTBlock(d_model, nhead, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x, key_padding_mask, attn_mask):
        x = self.embedding(x)  # 嵌入层
        for block in self.GPT_blocks:
            x = block(x, key_padding_mask, attn_mask)
        x = self.fc(x)
        return x
