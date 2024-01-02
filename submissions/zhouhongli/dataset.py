import json
from collections import Counter

import torch
from torch.utils.data import Dataset


class PoemDataset(Dataset):
    def __init__(self, data_path, vocab_path, is_train):
        self.is_train = is_train
        self.data = self.load_data(data_path)
        self.vocab_path = vocab_path
        if is_train:
            self.vocab_size, self.word2idx, self.idx2word = self.build_vocab()
            self.save_vocab(vocab_path)
        else:
            self.vocab_size, self.word2idx, self.idx2word= self.load_vocab(vocab_path)
            # json 中的 key 是 str 不是 int，所以这里转化一下
            self.idx2word = {int(key): value for key, value in self.idx2word.items()}

    def load_vocab(self, vocab_path):
        with open(vocab_path, 'r', encoding="utf-8") as f:
            vocab = json.load(f)
        return vocab['vocab_size'], vocab['word2idx'], vocab['idx2word']

    def load_data(self, data_path):
        with open(data_path, 'r', encoding="utf-8") as f:
            data = [json.loads(line) for line in f.readlines()]
        return data

    def build_vocab(self):
        # 构建词汇表

        content_text = [poem['content'] for poem in self.data]
        content_text_combined = '|'.join(content_text)
        # print(content_text_combined)
        content_words = ''.join(content_text_combined.split('|'))
        # print(content_words)

        keywords_text = [poem['keywords'] for poem in self.data]
        keywords_text_combined = ' '.join(keywords_text)
        keywords_words = ''.join(keywords_text_combined.split(' '))

        title_text = [poem['title'] for poem in self.data]
        title_words = ''.join(title_text)

        all_words = ''.join([content_words, keywords_words, title_words])

        # 统计整个训练集中的不同单词种类
        word_counts = Counter(all_words)
        unique_words = list(word_counts.keys())
        # print(unique_words)
        # print(word_counts)

        # 根据训练集中不同单词的数量确定词汇表大小
        vocab_size = len(unique_words)

        sorted_words = [word for word, _ in word_counts.most_common(vocab_size)]
        sorted_words = ['<pad>', '<bos>', '<eos>', ' ', '<unk>', '标题', '关键词', '诗歌', '|', '，', '五言', '七言',
                        '：'] + sorted_words

        vocab_size = len(sorted_words)
        # print(len(sorted_words))

        word2idx = {word: idx for idx, word in enumerate(sorted_words)}
        # print(word2idx)
        idx2word = {idx: word for word, idx in word2idx.items()}
        return vocab_size, word2idx, idx2word

    def save_vocab(self, vocab_path):
        vocab = {
            'vocab_size': self.vocab_size,
            'word2idx': self.word2idx,
            'idx2word': self.idx2word
        }
        with open(vocab_path, 'w', encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        poem = self.data[idx]

        title = poem['title']
        keywords = poem['keywords']
        content = poem['content']

        # 格式化
        # input_str = f"<bos>标题：{title}，关键词：五（七）言 {keywords}，诗歌："
        # target_str = f"{content}<eos>"

        input_indices = [self.word2idx.get('<bos>'), self.word2idx.get('标题'), self.word2idx.get('：')]
        # 分词并转换为索引
        for word in title:
            index = self.word2idx.get(word, self.word2idx['<unk>'])
            input_indices.append(index)
        input_indices.append(self.word2idx.get(' '))
        input_indices.append(self.word2idx.get('关键词'))
        input_indices.append(self.word2idx.get('：'))

        len_content = len(content)
        # 可能是因为训练集数据分布不均，七言诗远远多于五言诗
        # 模型架构又比较简单，所以当前模型无法很好地根据提示生成五言或七言
        if len_content == 23:
            input_indices.append(self.word2idx.get('五言'))
            input_indices.append(self.word2idx.get(' '))
        elif len_content == 31:
            input_indices.append(self.word2idx.get('七言'))
            input_indices.append(self.word2idx.get(' '))
        for word in keywords:
            index = self.word2idx.get(word, self.word2idx['<unk>'])
            input_indices.append(index)
        input_indices.append(self.word2idx.get(' '))
        input_indices.append(self.word2idx.get('诗歌'))
        input_indices.append(self.word2idx.get('：'))

        target_indices = []
        if self.is_train:
            target_indices += [self.word2idx['<pad>']] * (len(input_indices) - 1)
            for word in content:
                index = self.word2idx.get(word, self.word2idx['<unk>'])
                input_indices.append(index)

        for word in content:
            index = self.word2idx.get(word, self.word2idx['<unk>'])
            target_indices.append(index)
        if self.is_train:
            target_indices.append(self.word2idx.get('<eos>'))

        input_len = len(input_indices)
        target_len = len(target_indices)

        # print(''.join([self.idx2word.get(id, '<unk>') for id in input_indices]))
        return input_indices, target_indices, input_len, target_len, len_content

    def padding_batch(self, batch):
        input_lens = [input_len for _, _, input_len, target_len, _ in batch]
        target_lens = [target_len for _, _, input_len, target_len, _ in batch]
        lens_content = [len_content for _, _, _, _, len_content in batch]


        input_max_len = max(input_lens)
        target_max_len = max(target_lens)
        max_len_content = max(lens_content)

        for input, target, input_len, target_len, _ in batch:
            input.extend([self.word2idx['<pad>']] * (input_max_len - input_len))
            target.extend([self.word2idx['<pad>']] * (target_max_len - target_len))

        # 发现输入过长或过短时输出效果都不太好，调整一下输入的长度发现输入过长或过短时输出效果都不太好，调整一下输入的长度
        if not self.is_train:
            for input, target, input_len, target_len, _ in batch:
                # 如果输入长度过短，就在前面加 <bos>
                if input_len < 35:
                    for _ in range(35 - input_len):
                        input.insert(1, self.word2idx['<bos>'])

                # 如果输入长度过长，就删去一部分标题
                if input_len > 37:
                    for _ in range(input_len - 37):
                        input.pop(3)

        inputs = torch.tensor([input for input, target, _, _, _ in batch], dtype=torch.long)
        targets = torch.tensor([target for input, target, _, _, _ in batch], dtype=torch.long)
        # print(inputs.shape)  # (batch_size, input_len)
        return inputs, targets, max_len_content
