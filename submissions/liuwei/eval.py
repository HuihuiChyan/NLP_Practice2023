import pandas as pd
import numpy as np
from utils import set_seeds, print_gpu_info
from train import Block
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


# 参数
batch_size = 1
block_size = 256 # 最大 context length
learning_rate = 2e-5
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
n_embd = 64
n_head = 8
# head_size = n_embd // n_head
n_layer = 10
dropout = 0.1
epochs = 10



set_seeds(43)


dev_data = pd.read_json('data/ccpc_valid_v1.0.json',lines=True)
test_data = pd.read_json('data/ccpc_test_v1.0.json',lines=True)

dev_data = dev_data[['title','keywords','content']]
test_data = test_data[['title','keywords','content']]


stoi = {}
itos = {}
with open('vocab.txt', 'r') as f:
    for i, line in enumerate(f.readlines()):
        line = line.rstrip('\n')
        itos[i] = line
        stoi[line] = i

vocab_size = len(stoi)

vocab = stoi.keys()
print('词表大小：', vocab_size)


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

        self.xs = [(data[i])[:-1] for i in range(0, len(data))] # 去掉最后一个token
        self.ys = [(data[i])[1:] for i in range(0, len(data))] # 去掉第一个token
    

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        x = torch.tensor(self.xs[item]) #, dtype=torch.float)
        y = torch.tensor(self.ys[item]) #, dtype=torch.float)
        # x, y = x.to(device), y.to(device) 
        # 只有密集的CPU tensors才能被固定
        # 在PyTorch中，pin_memory操作是用来将数据预加载到固定（pinned）内存中的，这样在将数据移动到GPU时可以加速数据传输。
        return x, y
    

dev_dataset = MyDataset(dev_data)
test_dataset = MyDataset(test_data)

dev_loader = DataLoader(dev_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0, pin_memory=True, drop_last=False)


def eval(loader):

    loss = []

    for step, (inputs, labels) in enumerate(loader):

        # 要查找的序列
        sequence = torch.tensor(encode("诗歌："))

        positions = None

        for i, row in enumerate(inputs): # 确定只有一行
            for j in range(len(row) - len(sequence) + 1):
                if torch.all(row[j:j+len(sequence)] == sequence):
                    positions = j + len(sequence)

        inputs = inputs[:, :positions] # input截断到诗歌前面

        # context = torch.zeros((1, 1), dtype=torch.long, device=device)
        idx, loss_i = model.generate_poem_eval(inputs.to(device), labels.to(device))

        if step < 5:
            print("Q: ", decode(np.array(inputs[0])))
            print("A: ", decode(np.array(idx[0, positions:].cpu())))
            # print("loss: ", loss_i)

        loss.append(loss_i.item())

    print("困惑度：", np.mean(loss))



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
            # loss = torch.exp((loss * weights.view(-1)).sum() / weights.sum()) # 取个指数--困惑度
            
        return idx, loss
    

# 加载最优模型
model = GPT()
model.to(device)
model.load_state_dict(torch.load('best_model/best_model.pth'))
print_gpu_info()

test_poem = ["<s>标题：咏史二十二首 其十六 隋文帝●●●格式：五言●●●关键词：贻谋 学术 侯 失●●●诗歌：",
                "<s>标题：别离情所钟十二章章四句送定叟弟之官严陵 其十二●●●格式：五言●●●关键词：短 吹我 朔风 送子●●●诗歌：",
                "<s>标题：席上分得时字送豫斋二首 其一●●●格式：五言●●●关键词：见月 客中 泽国 登楼●●●诗歌：",
                "<s>标题：芙蓉●●●格式：五言●●●关键词：云锦 南州 青松 芙蓉●●●诗歌：",
                "<s>标题：羊城八景 其八 扶桑浴日●●●格式：五言●●●关键词：远空 东溟 金轮 洗出●●●诗歌：",
                "<s>标题：山居书怀●●●格式：七言●●●关键词：薜萝 云烟 青松 寒泉●●●诗歌：",
                "<s>标题：阳城●●●格式：七言●●●关键词：裂麻 谁云 抗疏 重轻●●●诗歌："]
for poem in test_poem:
    poem_new = model.generate_poem(torch.tensor(encode(poem)).reshape(1, -1).to(device))
    print(decode(np.array(poem_new[0].cpu())))


eval(dev_loader)

eval(test_loader)


