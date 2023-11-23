import pandas as pd
import numpy as np
from utils import set_seeds, print_gpu_info
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

 
# 参数
batch_size = 64
block_size = 256 # 最大 context length
learning_rate = 2e-5
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
n_embd = 64
n_head = 8
# head_size = n_embd // n_head
n_layer = 10
dropout = 0.1
epochs = 10


set_seeds(42) # 固定随机种子


class MyDataset(Dataset):

    def __init__(self, df):

        def geshi(x):
            if x.find('|')==5:
                return '五言'
            elif x.find('|')==7:
                return '七言'
            else:
                return '五言'

        df_text = df.apply(lambda row: '<s>标题：' + row['title'] + '●●●格式：' + geshi(row['content']) + '●●●关键词：' + row['keywords'] + '●●●诗歌：' + row['content'] + '</s>', axis=1)
        data = df_text.apply(lambda x: encode(x))

        self.xs = [self.pad_sequence((data[i])[:-1]) for i in range(0, len(data))] # 去掉最后一个token并填充
        self.ys = [self.pad_sequence((data[i])[1:]) for i in range(0, len(data))] # 去掉第一个token并填充
    
    def pad_sequence(self, seq):
        # 如果序列长度小于block_size，就在末尾添加<mask>
        if len(seq) < block_size:
            seq += [stoi['<mask>']] * (block_size - len(seq))
        return seq[:block_size]


    # def __init__(self, df):

    #     df_text=df.apply(lambda row: '<s>标题：' + row['title'] + ' 关键词：' + row['keywords'] + ' 诗歌：' + row['content'] + '</s>', axis=1)

    #     data = []
    #     for i in df_text:
    #         data += encode(i)

    #     self.xs = [data[i:i+block_size] for i in range(0, len(data)-block_size, block_size)] # 按照block_size进行分组，会舍弃掉最后一些token
    #     self.ys = [data[i+1:i+1+block_size] for i in range(0, len(data)-block_size, block_size)] # 需要 + 1

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        x = torch.tensor(self.xs[item]) #, dtype=torch.float)
        y = torch.tensor(self.ys[item]) #, dtype=torch.float)
        # x, y = x.to(device), y.to(device) 
        # 只有密集的CPU tensors才能被固定
        # 在PyTorch中，pin_memory操作是用来将数据预加载到固定（pinned）内存中的，这样在将数据移动到GPU时可以加速数据传输。
        return x, y
    

@torch.no_grad()
def estimate_loss(): # 后续再优化，应该和训练放一起
    out = {}
    model.eval()
    for split in ['train', 'dev']:
        if split == 'train':
            loader = train_loader
        elif split == 'dev':
            loader = dev_loader

        losses = []
        for step, (inputs, labels) in enumerate(loader):
            if step == 5:
                break
            logits, loss, _ = model(inputs.to(device), labels.to(device))
            losses.append(loss.item())
            # 因为要计算每一组数据的loss，所以才需要mean，只计算一句话的话不需要mean
        # out[split] = np.mean(np.array(torch.exp(torch.tensor(losses))))
        out[split] = np.mean(np.array(torch.tensor(losses))) # 已经计算过指数了
    model.train()
    return out


class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)

        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    

class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # sin和cos组成的的position_embedding
        position_embedding = torch.zeros(block_size, n_embd)
        for pos in range(block_size):
            for i in range(0, n_embd, 2):
                position_embedding[pos, i] = np.sin(pos/(10000**((2 * i) / n_embd)))
                if i+1 < n_embd:
                    position_embedding[pos, i+1] = np.cos(pos/(10000**((2 * i) / n_embd)))
        self.position_embedding_table = torch.tensor(position_embedding, dtype=torch.float32)
        # 更推荐：self.position_embedding_table = position_embedding.clone().detach()

        # self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 因为forward时各维度可能小于init里面定义的维度，故进行索引
        tok_emb = self.token_embedding_table(idx) # (B,T,C)  c是n_embd
        # pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        pos_emb = (self.position_embedding_table)[:T]   # (T,C)
        x = tok_emb + pos_emb.clone().detach().to(device)  # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
            loss_sum = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
  
            loss = F.cross_entropy(logits, targets, reduction='none')
            loss = loss.view(B, T)
            targets = targets.view(B, T)
            weights = torch.zeros_like(loss).float()
            for i in range(targets.shape[0]):
                for j in range(targets.shape[1] - 2):
                    if targets[i, j:j+3].tolist() == torch.tensor(encode("诗歌：")).tolist():
                        # 从此位置开始计算损失，直到遇到mask
                        for k in range(j+3, targets.shape[1]):
                            if (targets[i, k]).item() == stoi['<mask>']: #出现mask则结束
                                break
                            weights[i, k] = 1
                        break
            loss = torch.exp((loss * weights).sum() / weights.sum())


            # loss = F.cross_entropy(logits, targets, ignore_index=2)
            loss_sum =  F.cross_entropy(logits, targets.view(B*T), reduction='none') # 为了在评估时只计算后面诗句的损失，故这里不对损失求mean

        return logits, loss, loss_sum

    def generate_poem(self, idx, max_new_tokens=block_size):  # 一次一个batch

        len_input = idx.shape[1]

        model.eval()
        with torch.no_grad():
            for _ in range(len_input, max_new_tokens):

                logits, _, _ = self(idx)
                
                logits = logits[0, -1, :] # (B=1, 1, C)

                probs = F.softmax(logits, dim=-1)

                idx_next = torch.multinomial(probs, num_samples=1) # (B=1, 1)

                idx = torch.cat((idx, idx_next.reshape(1,1)), dim=1) # (B=1, T+1)

                if idx_next in [0, 1]:
                    break
            
        return idx
    
    def generate_poem_eval(self, idx, tgt, max_new_tokens=block_size):  # 一次一个batch


        len_input = idx.shape[1]

        max_new_tokens = tgt.shape[1]

        model.eval()
        with torch.no_grad():
            for _ in range(len_input, max_new_tokens):

                logits, _, _ = self(idx)
                
                logits = logits[0, -1, :] # (B=1, 1, C)

                probs = F.softmax(logits, dim=-1)

                idx_next = torch.multinomial(probs, num_samples=1) # (B=1, 1)

                idx = torch.cat((idx, idx_next.reshape(1,1)), dim=1) # (B=1, T+1)

                # if idx_next in [0, 1]:
                #     break
            

            _, loss, _ = self(idx, tgt)

            # weights = torch.zeros_like(tgt).float()
            # weights[:, len_input:] = 1 # len_input后面的为诗句
            # loss = torch.exp((loss_sum * weights.view(-1)).sum() / weights.sum()) # 取个指数--困惑度
            
        return idx, loss
    

if __name__=="__main__":


    train_data = pd.read_json('data/ccpc_train_v1.0.json',lines=True)
    dev_data = pd.read_json('data/ccpc_valid_v1.0.json',lines=True)

    train_data = train_data[['title','keywords','content']]
    dev_data = dev_data[['title','keywords','content']]

    text = ''

    for i in train_data.index:
        for j in train_data.columns:
            text += train_data.loc[i,j]

    text = text.replace("\n",'')

    chars = sorted(list(set(text)))

    stoi = { ch:i+4 for i,ch in enumerate(chars) } # 字符转换为token
    itos = { i+4:ch for i,ch in enumerate(chars) } # token转换为字符

    # 添加特殊标识符
    stoi['<s>'] = 0
    stoi['</s>'] = 1
    stoi['<unk>'] = 2
    stoi['<mask>'] = 3
    itos[0] = '<s>'
    itos[1] = '</s>'
    itos[2] = '<unk>'
    itos[3] = '<mask>'

    stoi = dict(sorted(stoi.items(), key=lambda x:x[1], reverse=False))
    itos = dict(sorted(itos.items(), key=lambda x:x[0], reverse=False))

    vocab_size = len(stoi)
    vocab = stoi.keys()

    print('词表大小：', vocab_size)
    print('一个epoch迭代次数：', int(train_data.shape[0]/batch_size))

    with open('vocab.txt', 'w') as f:
        for i in vocab:
            f.write(i+'\n')

    def encode(s, max_len=5):
        i = 0
        encode = []
        while i <= len(s):
            max_len_i = max_len
            while max_len_i >= 0:
                if max_len_i == 0:
                    encode.append(stoi['<unk>']) # 未知token填入unk
                    i += 1
                    break
                if s[i:i+max_len_i] in vocab:
                    encode.append(stoi[s[i:i+max_len_i]])
                    i += max_len_i
                    break

                max_len_i -= 1
        
        return encode

    decode = lambda l: ''.join([itos[i] for i in l])

    # print(encode("<s>\ue000123请你写首诗</s>"))
    # print(decode(encode("<s>\ue000123请你写首诗</s>")))


    train_dataset = MyDataset(train_data)
    dev_dataset = MyDataset(dev_data)

    train_loader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0, pin_memory=True, drop_last=True)
    dev_loader = DataLoader(dev_dataset,
                                batch_size=batch_size * 2,
                                shuffle=True,
                                num_workers=0, pin_memory=True, drop_last=False)


    print_gpu_info()
    model = GPT()
    # print(model)
    model.to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        print_gpu_info()

        for step, (inputs, labels) in enumerate(train_loader):
            
            if step%100 == 0:

                print_gpu_info()
                start_time = datetime.datetime.now()
                losses = estimate_loss()
                print(f"epoch {epoch} and step {step}: train loss {losses['train']:.2f}, val loss {losses['dev']:.2f}")
                end_time = datetime.datetime.now()
                print("运行了", (end_time - start_time).total_seconds(), "秒")

                if losses['dev']<best_val_loss:
                    best_val_loss = losses['dev']  
                    # print(f'已保存最优模型，验证集loss: {round(best_val_loss, 2)}')
                    torch.save(model.state_dict(), f'best_model/best_model.pth')

                print("\n")
            # else:
            #     print(f"epoch {epoch} and step {step}")

            # evaluate the loss
            logits, loss, _ = model(inputs.to(device), labels.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    print("所有epoch运行完毕！！！")
    print_gpu_info()


    # 加载最优模型
    model = GPT()
    model.to(device)
    model.load_state_dict(torch.load('best_model/best_model.pth'))
    print_gpu_info()

    test_poem = ["<s>标题：咏史二十二首 其十六 隋文帝●●●格式：五言●●●关键词：贻谋 学术 侯 失●●●诗歌：",
                "<s>标题：别离情所钟十二章章四句送定叟弟之官严陵 其十二●●●格式：五言●●●关键词：短 吹我 朔风 送子●●●诗歌：",
                "<s>标题：席上分得时字送豫斋二首 其一●●●格式：五言●●●关键词：见月 客中 泽国 登楼●●●诗歌：",
                "<s>标题：芙蓉●●●格式：五言●●●关键词：云锦 南州 青松 芙蓉●●●诗歌：",
                "<s>标题：羊城八景 其八 扶桑浴日●●●格式：五言●●●关键词：远空 东溟 金轮 洗出●●●诗歌："]

    for poem in test_poem:
        poem_new = model.generate_poem(torch.tensor(encode(poem)).reshape(1, -1).to(device))
        print(decode(np.array(poem_new[0].cpu())))



 