import argparse
import json
from tqdm import tqdm

#生成词表
def main():
    in_path='./data/train.json'
    out_path='./data/vocab.txt'
    words={}
    with open(in_path, 'r', encoding="utf-8") as f:
        lines=json.load(f)
    for i,line in enumerate(tqdm(lines)):
        for item in line:
            if item in words.keys():
                words[item] = words[item]+1
            else:
                words[item]=1

    words = sorted(words.items(), key=lambda x: x[1], reverse=True)
    words = dict(words)
    vocab = list(words.keys())
    pre = ['<PAD>', '<UNK>']
    vocab = pre + vocab
    with open(out_path, 'w', encoding="utf-8") as f:
        for word in vocab:
            f.write(word + '\n')





if __name__ == "__main__":
    main()