import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PoemDataset
from gpt import GPTGenerator

# GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_path = './vocab.json'
data_path = './data/ccpc_train_v1.0.json'

# 超参数
batch_size = 32
d_model = 768
nhead = 4
num_layers = 12
d_ff = 1024
dropout = 0.1
num_epochs = 40  # 总共的训练轮数
learning_rate_1 = 1e-4  # 前 30 个 epoch 的学习率
learning_rate_2 = 1e-5  # 后 10 个 epoch 的学习率



# 加载数据集
poem_dataset = PoemDataset(data_path, vocab_path, True)
vocab_size = poem_dataset.vocab_size

train_loader = DataLoader(poem_dataset, batch_size, shuffle=True, drop_last=True, collate_fn=poem_dataset.padding_batch)

# 初始化模型
model = GPTGenerator(vocab_size, d_model, nhead, num_layers, d_ff, dropout, device)
model.to(device)

# 定义优化器和学习率调度器
optimizer = Adam(model.parameters(), lr=learning_rate_1)
scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: learning_rate_2 if epoch >= 30 else learning_rate_1)
# 定义损失函数
# 如果使用 CrossEntropyLoss()，不需要再加 softmax 了
criterion = nn.CrossEntropyLoss(ignore_index=0)

# 梯度裁剪阈值
clip_value = 1


def create_attention_mask(batch_size, seq_len, num_heads):
    # 生成上三角矩阵
    mask = torch.triu(torch.ones(seq_len, seq_len))

    # 复制到每个样本和每个头
    mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, num_heads, -1, -1)

    return mask.bool()


# 训练过程
for epoch in range(num_epochs):
    total_loss = 0
    tqdm_batch_iterator = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', dynamic_ncols=True)
    for input, target, _ in tqdm_batch_iterator:
        # with autograd.detect_anomaly():
        # print(input.shape)
        attn_mask = create_attention_mask(batch_size, input.size(1), nhead).to(device)
        # 生成 key_padding_mask
        key_padding_mask = (input == 0)
        key_padding_mask = key_padding_mask.to(device)
        # print(key_padding_mask)

        input = input.to(device)
        # 将数据传递给模型，并传递掩码
        output = model(input, key_padding_mask, attn_mask)

        # print(output.shape)  # batch_size * input_len * vocab_size
        output = output.view(-1, vocab_size).to(device)
        target = target.view(-1).to(device)
        # print(target.shape)

        # 在计算损失之前检查是否为 nan
        if torch.isnan(output).any():
            print("Model output contains NaN values!")
            print("Output:", output)

        # 计算损失
        loss = criterion(output, target)
        loss.to(device)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        total_loss += loss.item()

        # 更新进度条
        tqdm_batch_iterator.set_postfix(loss=total_loss / (tqdm_batch_iterator.n + 1), refresh=True)

    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}')

    # 更新学习率
    scheduler.step()

# 保存模型
torch.save(model.state_dict(), 'gpt_model.pth')
