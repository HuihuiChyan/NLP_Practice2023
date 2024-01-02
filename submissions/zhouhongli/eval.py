import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PoemDataset
from gpt import GPTGenerator
from tqdm import tqdm

# 超参数
batch_size = 1
d_model = 768
nhead = 4
num_layers = 12
d_ff = 1024
dropout = 0.1

sf = nn.Softmax(dim=-1)


def create_attention_mask(batch_size, seq_len, num_heads):
    # 生成上三角矩阵
    mask = torch.triu(torch.ones(seq_len, seq_len))

    # 复制到每个样本和每个头
    mask = mask.unsqueeze(0).unsqueeze(1).expand(batch_size, num_heads, -1, -1)

    return mask.bool()


def generate_ids(poem_dataset, input_ids, len_content):

    vocab_size = poem_dataset.vocab_size

    # 初始化模型
    model = GPTGenerator(vocab_size, d_model, nhead, num_layers, d_ff, dropout, device)
    model.load_state_dict(torch.load('gpt_model.pth'))
    model.to(device)
    model.eval()

    with torch.no_grad():
        current_ids = []
        input_ids = input_ids.tolist()

        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        logits_list = []  # 用于保存每次预测的 logits
        for i in range(len_content):

            key_padding_mask = (input_tensor == 0)
            key_padding_mask = key_padding_mask.to(device)

            attn_mask = create_attention_mask(1, input_tensor.size(1), 4).to(device)

            output = model(input_tensor, key_padding_mask, attn_mask)

            output = output[:, -1, :]
            logits_list.append(output)  # 将当前时刻的 logits 加入列表中

            next_logit = sf(output)  # 对输出进行 softmax
            next_token = torch.multinomial(next_logit, 1).item()  # 从多项分布中随机抽取一个词

            input_ids.append(next_token)

            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
            current_ids.append(next_token)

        logits = torch.stack(logits_list, dim=1)  # 将列表中的 logits 拼接成张量，维度为 (batch_size, len_content, vocab_size)

    return logits

def calculate_perplexity(criterion, poem_dataset):
    total_loss = 0.0
    length = 0

    data_loader = DataLoader(poem_dataset, batch_size, shuffle=False, drop_last=True,
                             collate_fn=poem_dataset.padding_batch)

    with torch.no_grad():
        for input, target, len_content in tqdm(data_loader, desc='Calculating Perplexity',
                                               leave=False):  # 添加 tqdm 包装器，用于显示进度条
            target = target.to(device)
            input = input.to(device)[0]

            logits = generate_ids(poem_dataset, input, len_content)

            # 传递 logits 给交叉熵损失函数进行计算
            loss = criterion(logits.view(-1, poem_dataset.vocab_size), target.view(-1))

            total_loss += loss.item()
            length += 1

            average_loss = total_loss / length
            perplexity = torch.exp(torch.tensor(average_loss)).item()
            print(average_loss)

    average_loss = total_loss / length
    perplexity = torch.exp(torch.tensor(average_loss)).item()

    return perplexity


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_path = './vocab.json'
    val_data_path = './data/ccpc_valid_v1.0.json'
    test_data_path = './data/ccpc_test_v1.0.json'

    # 加载验证集和测试集的数据
    val_dataset = PoemDataset(val_data_path, vocab_path, False)
    test_dataset = PoemDataset(test_data_path, vocab_path, False)


    # 定义损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # 计算验证集和测试集的 perplexity
    val_perplexity = calculate_perplexity(criterion, val_dataset)
    test_perplexity = calculate_perplexity(criterion, test_dataset)

    print(f'Validation Perplexity: {val_perplexity:.4f}')
    print(f'Test Perplexity: {test_perplexity:.4f}')
