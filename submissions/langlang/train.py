import json

import torch
import torch.nn as nn
from config import Config
from gpt import GPT
from torch.utils.data import DataLoader, Dataset


class PoemDataSet(Dataset):
    """
        诗歌数据集，便于使用pytorch中的DataLoader

        必须实现__getitem__、__len__等函数
    """

    def __init__(self, poemdata):
        self.poemdata = poemdata

    def __len__(self):
        return len(self.poemdata)

    def __getitem__(self, item):
        data_item = self.poemdata[item]
        idx = data_item.index(5) + 2
        train_input = data_item[:-1]
        train_output = data_item[1:]
        poem_output = data_item[1:]
        for i in range(idx):
            poem_output[i] = 0
        return_obj = {
            "train_input": train_input,
            "train_output": train_output,
            "poem_output": poem_output
        }
        return return_obj

    def collate_fn(self, batch):
        """
        将同一批batch的数据处理成可以等长的tensor
        :param batch: 一个批次的数据
        :return: 处理后的tensor
        """
        # 获取数据的最大长度 将不足的部分填充为<pad>
        input_max_len = -1
        output_max_len = -1
        poem_max_len = -1
        for data in batch:
            input_max_len = max(input_max_len, len(data['train_input']))
            output_max_len = max(output_max_len, len(data['train_output']))
            poem_max_len = max(poem_max_len, len(data['poem_output']))

        # 此处默认<pad>映射为0
        for data in batch:
            data['train_input'].extend([0] * (input_max_len - len(data['train_input'])))
            data['train_output'].extend([0] * (output_max_len - len(data['train_output'])))
            data['poem_output'].extend([0] * (poem_max_len - len(data['poem_output'])))

        # 将原始数据转化为tensor
        decoder_input = torch.tensor([data['train_input'] for data in batch], dtype=torch.long)
        decoder_output = torch.tensor([data['train_output'] for data in batch], dtype=torch.long)
        poem = torch.tensor([data['poem_output'] for data in batch], dtype=torch.long)

        return decoder_input, decoder_output, poem


def make_data(datas):
    """
    处理文件中预先写好的数据 将数据添加<sep>分隔符
    :param datas: 数据list
    :return: 处理后的数据
    """
    train_datas = []
    for data in datas:
        train_data = [i if i != '\t' else "<sep>" for i in data] + ['<sep>']
        train_datas.append(train_data)

    return train_datas


def write_traindata_tofile(train_url, save_url):
    """
    读取训练数据集 按照指定格式 写入指定文件中
    :param train_url: 训练数据集的路径
    :param save_url: 指定文件路径
    :return: None
    """
    results = []
    # 将源数据按照指定格式读取至列表中
    with open(train_url, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line_str = line.rstrip('\n')
            obj_json = json.loads(line_str)
            content = obj_json['content']  # 23个字符的为五言 否则为七言
            title = obj_json['title']
            keywords = obj_json['keywords']
            piece_data = list()
            piece_data.append("标题")
            piece_data.append("：")
            for char in title:
                piece_data.append(char)
            piece_data.append("***")
            piece_data.append("关键词")
            piece_data.append("：")
            if len(content) == 23:
                piece_data.append("五言")
            else:
                piece_data.append("七言")
            piece_data.append("\t")
            for char in keywords:
                piece_data.append(char)
            piece_data.append("***")
            piece_data.append("诗歌")
            piece_data.append("：")
            piece_data.append("\t")
            for char in content:
                piece_data.append(char)
            results.append(piece_data)
            line = f.readline()
    f.close()
    with open(save_url, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
    f.close()


def train(model, data_loader):
    """
    训练模型
    :param model: 待训练的模型--torch.nn.Module
    :param data_loader: 数据加载器--torch.utils.data.DataLoader
    :return: None
    """
    cross = nn.CrossEntropyLoss(ignore_index=0, reduction='sum').to(config.device)
    # ignore_index 不计算损失的值 默认为-100
    # reduction 计算方式 默认为"None" 可选"mean"和"sum"

    opti = torch.optim.Adam(params=model.parameters(), lr=config.learning_rate)
    # params 训练过程中需要优化的参数
    # lr 学习率

    old_loss = 100

    for epoch in range(config.epoch):  # 每一轮的训练
        print(f"第{epoch + 1}轮训练开始。")
        train_loss = train_step(model, data_loader, opti, cross, config.max_norm)  # 每一步的训练
        # if train_loss>old_loss:
        #     print("训练结束")
        #     break
        # torch.save(model.state_dict(),
        # f"./model/{config.model_pre}_{config.num_layers}_{config.num_heads}_{epoch+1}.pt")
        torch.save(model.state_dict(), "GPT_last.pt")
        print(f'本轮的训练损失:{train_loss:.4f}')
        # print(f'本轮的训练损失:{train_loss:.4f},上一轮的训练损失:{old_loss:.4f}')
        # old_loss = train_loss


def train_step(model, data_loader, opti, cross, max_norm=1):
    """
    训练某一轮数据
    :param model: 训练模型--torch.nn.Module
    :param data_loader: 数据加载器--torch.utils.Data.DataLoader
    :param opti: 优化器--torch.opti.Adam
    :param cross: 交叉熵--torch.nn.CrossEntropyLoss
    :param max_norm: 最大的范数
    :return: 这批数据的在本轮次训练的损失
    """
    model.train()
    epoch_loss = 0

    for (dec_inputs, dec_outputs, poem_outputs) in data_loader:
        opti.zero_grad()

        dec_inputs, dec_outputs, poem_outputs = dec_inputs.to(config.device), dec_outputs.to(
            config.device), poem_outputs.to(config.device)
        mask = poem_outputs == dec_outputs
        outputs, _ = model(dec_inputs)  # 获得模型输出
        dec_outputs *= mask

        loss = cross(outputs, dec_outputs.view(-1))  # 计算损失 其中tensor.view(-1)是将输出压成一行 进行计算

        num = dec_outputs.ne(0).long().sum().item()  # 判断输出是否等于0（填充符<pad>)，计算非0的数目，转化为python数据类型
        loss = loss / num
        epoch_loss += loss.item()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        opti.step()
    return epoch_loss / len(data_loader)


if __name__ == "__main__":
    config = Config()
    write_traindata_tofile(config.train_url, config.handle_train_url)

    with open(config.handle_train_url, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    f.close()
    train_data = make_data(train_data)
    # print(train_data)

    with open(config.dict_url, 'r', encoding='utf-8') as f:
        all_data = json.load(f)
    f.close()
    char2index = all_data['char2index']
    index2char = all_data['index2char']
    index2char.append("\n")
    char2index["\n"] = len(index2char) - 1

    train_data_vocab = [[char2index[word] for word in line] for line in train_data]
    # print(train_data_vocab)

    train_dataset = PoemDataSet(train_data_vocab)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.bacth_size,
                                  collate_fn=train_dataset.collate_fn)

    print(f"本次训练使用的设备是{config.device}")

    gpt_model = GPT().to(config.device)
    # model = nn.DataParallel(gpt_model,device_ids = [0,2,3])
    train(gpt_model, train_dataloader)
    # train(model,train_dataloader)
