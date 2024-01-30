# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import torch
from config import Config
import json

from train import write_traindata_tofile


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    config = Config()
    write_traindata_tofile(config.train_url, config.handle_train_url)
    write_traindata_tofile(config.test_url, config.handle_test_url)
    # result = np.triu(np.ones([2,3]), k=1)
    # print(result)
    # tensor1 = torch.tensor([1, 2, 3, 3, 5, 6])
    # tensor2 = torch.tensor([6, 5, 4, 3, 2, 1])
    # print(torch.gt(tensor1, 0))
    # print(torch.gt(tensor2, tensor1))
    # seq_k = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    # print((seq_k.size()))
    # seq_k = seq_k.transpose(0,1)
    # print(seq_k,seq_k.size())
    # pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], True is masked
    # print(pad_attn_mask,pad_attn_mask.size())
    # print(pad_attn_mask.expand(4, 3, 3))  # [batch_size, len_q, len_k]
    pad_attn_mask = torch.tensor([[[1,2],[2,3],[3,4]],[[1,2],[2,3],[3,4]],[[1,2],[2,3],[3,4]],[[1,2],[2,3],[3,4]]])
    temp = pad_attn_mask.view(-1,pad_attn_mask.size(-1))
    print(temp,temp.size())
    # print_hi('PyCharm')
    # config = Config()
    # print(config.device_list)
    # device_list = torch.cuda.device_count()
    # print(device_list)
    # with open("./data/dict.json", 'r', encoding='utf-8') as f1:
    #     data1 = json.load(f1)
    # with open("./data/dict_datas.json", "r", encoding='utf-8') as f2:
    #     data2 = json.load(f2)
    #
    # index2char = data1['index2char']
    # id2word = data2['id2word']
    # word2id = data2['word2id']
    # char2index = data1['char2index']
    #
    # print(char2index['|'])
    #
    # for char in word2id:
    #     if word2id[char] == 7:
    #         print("???")
    # for char in index2char:
    #     if char not in id2word:
    #         print(char)
    # print(id2word[7] == id2word[5376])
    # print(index2char[389] == index2char[7])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
