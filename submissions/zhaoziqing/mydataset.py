"""
准备数据集
"""
import json

from torch.utils.data import DataLoader,Dataset
import torch
import config
pad_id = 0
unk_id = 1
sep_id = 2


class MySequence:
    def __init__(self):
        with open(config.vocab_path, 'r', encoding="utf-8") as f:
            data = [line.strip() for line in f.readlines()]
        data.append("\n")
        self.dict = {}
        for i, item in enumerate(data):
            self.dict[item] = i
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, x, y):
        if len(x) > config.max_len:
            temp = x.index("关键词") - 2
            for i in range(len(x) - config.max_len):
                x.pop(temp)
                y.pop(temp - 1)
                temp = temp - 1
        else:
            x.extend(["<PAD>"] * (config.max_len - len(x)))
            y.extend(["<PAD>"] * (config.max_len - len(y)))

        x = [self.dict.get(i, unk_id) for i in x]
        y = [self.dict.get(i, unk_id) for i in y]
        return x, y


    def __len__(self):
        return len(self.dict)


class MyDataset(Dataset):
    def __init__(self, path):
        with open(path, 'r', encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][:-1]
        y = self.data[idx][1:]
        return x, y


def data_count():
    with open(config.train_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    count = len(data)
    min = 1000
    max = 0
    long_list=[]
    total = 0
    for i, item in enumerate(data):
        t = len(item)
        if t > 250:
            long_list.append(i)
        if t < min:
            min = t
        if t > max:
            max = t
        total += t
    mean = total / count
    print("total:%d min:%d max:%d mean:%f" % (total, min, max, mean))
    print(len(long_list))


ns = MySequence()


def collate_fn(batch):
    input, output, poem = [], [], []
    for x, y in batch:
        x, y = ns.transform(x, y)
        z = y.copy()
        index = x.index(sep_id)
        for i in range(index+1):
            z[i] = pad_id
        input.append(x)
        output.append(y)
        poem.append(z)
    input = torch.LongTensor(input)
    output = torch.LongTensor(output)
    poem = torch.LongTensor(poem)
    return input, output, poem


def get_dataloader(path, batch_size):
    return DataLoader(MyDataset(path), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


if __name__ == '__main__':
    loader = get_dataloader(config.train_path, batch_size=10)
    for idx, (input,target,poem) in enumerate(loader):
        print(idx)
        print(input.shape)
        print(target.shape)
        break








