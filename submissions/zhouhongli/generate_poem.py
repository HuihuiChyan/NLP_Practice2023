import torch
import torch.nn as nn

from dataset import PoemDataset
from gpt import GPTGenerator

# GPU 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_path = './vocab.json'

# 超参数
batch_size = 32
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


def generate_poem(model, input_ids):
    model.eval()
    # print(input_ids)

    with torch.no_grad():
        current_text = ''.join([poem_dataset.idx2word.get(id, '<unk>') for id in input_ids])
        # print(current_text)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        # print(input_tensor.shape)  # 1 * input_len
        for i in range(31):
            key_padding_mask = (input_tensor == 0)
            key_padding_mask = key_padding_mask.to(device)

            attn_mask = create_attention_mask(1, input_tensor.size(1), nhead).to(device)

            output = model(input_tensor, key_padding_mask, attn_mask)
            # print(output.shape)
            output = output[:, -1, :]

            next_logit = sf(output)  # 对输出进行 softmax
            next_token = torch.multinomial(next_logit, 1).item()  # 从多项分布中随机抽取一个词
            if next_token == poem_dataset.word2idx['<eos>']:
                break

            input_ids.append(next_token)
            # print(input_ids)
            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
            current_text += poem_dataset.idx2word[next_token]

    return current_text


if __name__ == "__main__":
    # 加载测试集数据
    poem_data_path = './data/ccpc_test_v1.0.json'
    poem_dataset = PoemDataset(poem_data_path, vocab_path, False)
    vocab_size = poem_dataset.vocab_size

    # 选择测试集中前 5 个样本
    for index in range(5):

        input_ids = poem_dataset[index][0]

        if len(input_ids) < 35:
            for _ in range(35 - len(input_ids)):
                input_ids.insert(1, poem_dataset.word2idx['<bos>'])

        if len(input_ids) > 37:
            for _ in range(len(input_ids) - 37):
                input_ids.pop(3)

        # print(len(input_ids))
        # for id in input_ids:
        #     print(type(id))  # int

        # 加载模型和生成诗歌
        model = GPTGenerator(vocab_size, d_model, nhead, num_layers, d_ff, dropout, device)
        model.load_state_dict(torch.load('gpt_model.pth'))
        model.to(device)

        generated_poem = generate_poem(model, input_ids)

        generated_poem = generated_poem.replace('<bos>', '')
        generated_poem = generated_poem.replace('诗歌：', '诗歌：\n')
        generated_poem = generated_poem.replace('，', ' ')
        print(generated_poem + '\n')
