import torch
from torch import nn
import pandas as pd
import math
from torch.utils.data import DataLoader,Dataset
import pdb

def Add_words(words,str):
    # 向列表中添加词
    # [words.append(x) for x in str if x not in words]
    for x in str:
        if x not in words:
            words.append(x)
    return words

def Build_dic_txt(data):
    # 构建词典，单字分词
    with open("dict.txt","w") as file:
        # 特殊词先写入字典文件
        file.write("<BOS>\n")
        file.write("<EOS>\n")
        file.write("<UNK>\n")
        file.write("题目\n")
        file.write("关键词\n")
        file.write("内容\n")
        file.write(",\n")
        file.write(":\n")
        file.write("<pad>\n")
        file.write("五言\n")
        file.write("七言\n")
         
        # 先将文本存储在list中,并去重
        words = []
        for poem in data.iterrows():
            title = poem[1].at['title']
            keywords = poem[1].at['keywords']
            # pdb.set_trace()
            content = poem[1].at['content']
            
            # 读入题目单词
            words = Add_words(words,title)
            
            # 读入关键词单词
            words = Add_words(words,keywords)
            
            # 读入内容单词
            words = Add_words(words,content)
        
        # 将常见词写入字典文件
        lenth = len(words)
        for i in range(lenth):
            file.write(words[i] + "\n")

def Read_dic(dic_pth):
    with open(dic_pth,"r") as file:
        idx = 0
        Word2id = {}
        Id2word = {}
        for line in file:
            line = line.rstrip('\n') # 去掉结尾换行符
            Word2id[line] = idx
            Id2word[idx] = line
            idx += 1
    return Word2id,Id2word

def tokenizer(sent,Word2id):
    # 对某一字符串进行序列化
    token_sen = []
    lenth = len(sent)
    for i in range(lenth):
        word = sent[i]
        if word in Word2id:
            token_sen.append(Word2id[word])
        else:
            token_sen.append(Word2id['<UNK>'])
    return token_sen

def Build_target(lenth,ig_idx):
    array = []
    for i in range(lenth):
        array.append(ig_idx)
    return array

def Train_tokenizer(sentence,word2id,id2word):
    # 分词器 输出input和target
    # ignore_idx
    ig_idx = 8
    # 句子开头要加入<BOS>
    inp = [0]
    target = []
    
    str_title = sentence['title']
    str_keywords = sentence['keywords']
    str_content = sentence['content']
    # 题目:
    inp.append(3)
    inp.append(7)
    inp += tokenizer(str_title,word2id)
    target += Build_target(len(str_title) + 2,ig_idx)

    # ,
    inp.append(6)
    
    # 关键词:
    inp.append(4)
    inp.append(7)
    # 加入五言七言
    lenth_content = len(str_content)
    if lenth_content == 23:
        # 五言
        inp.append(9)
        inp.append(word2id[' '])
    else:
        # 七言
        inp.append(10)
        inp.append(word2id[' ']) 
    inp += tokenizer(str_keywords,word2id)
    target += Build_target(len(str_keywords) + 5,ig_idx)

    # ,
    inp.append(6)
    
    # 内容:
    inp.append(5)
    inp.append(7)
    target += Build_target(3,ig_idx)

    m = tokenizer(str_content,word2id)
    inp += m
    target += m

    #对于目标要在句子结尾加上'<EOS>'
    target.append(1)
    
    return inp,target

def Generate_tokenizer(sentence,word2id):
    # 推理过程中的序列化函数，处理数据，将数据分成“题目:xxx,关键词:xxx,内容:“、"xxxxx|xxxxx|xxxxx|xxxxx"，一个作为自回归的输入，一个作为target
    # ignore_idx
    ig_idx = 8
    # 句子开头要加入<BOS>
    inp = [0]
    target = []
    
    str_title = sentence['title']
    str_keywords = sentence['keywords']
    str_content = sentence['content']
    # 题目:
    inp.append(3)
    inp.append(7)
    inp += tokenizer(str_title,word2id)
    # target += Build_target(len(str_title) + 2,ig_idx)

    # ,
    inp.append(6)
    
    # 关键词:
    inp.append(4)
    inp.append(7)
    # 加入五言七言
    lenth_content = len(str_content)
    if lenth_content == 23:
        # 五言
        inp.append(9)
        inp.append(word2id[' '])
    else:
        # 七言
        inp.append(10)
        inp.append(word2id[' ']) 
    inp += tokenizer(str_keywords,word2id)
    # target += Build_target(len(str_keywords) + 5,ig_idx)

    # ,
    inp.append(6)
    
    # 内容:
    inp.append(5)
    inp.append(7)
    # target += Build_target(3,ig_idx)

    target += tokenizer(str_content,word2id)
    

    # #对于目标要在句子结尾加上'<EOS>'
    # target.append(1)
    return inp,target,lenth_content
    # return inp,target,len(str)
    
class MyDataset(Dataset):
    # 将训练集封装成Dataloader的参数类型
    def __init__(self,data,Word2id,Id2word):
        self.data = data
        self.Word2id = Word2id
        self.Id2word = Id2word
    
    def __getitem__(self, index):
        inp,target = Train_tokenizer(self.data.loc[index],self.Word2id,self.Id2word) 
        inp_len = len(inp)
        target_len = len(target)
        return {'input':inp,'target':target,'input_len':inp_len,'target_len':target_len}
    
    def __len__(self):
        return len(self.data)
    
    def padding_batch(self,batch):
        input_lens = [d['input_len'] for d in batch]
        target_lens = [d['target_len'] for d in batch]
        
        input_max_len = max(input_lens)
        target_max_len = max(target_lens)    
        
        for d in batch:
            d['input'].extend([self.Word2id['<pad>']] * (input_max_len - d['input_len']))
            d['target'].extend([self.Word2id['<pad>']] * (target_max_len - d['target_len'])) 
        inputs = torch.tensor([d['input'] for d in batch],dtype=torch.long)
        outputs = torch.tensor([d['target'] for d in batch],dtype=torch.long)
        return inputs,outputs
    
class Mytest_Dataset(Dataset):
    # 将测试集封装成Dataloader的参数类型
    def __init__(self,data,Word2id,Id2word):
        self.data = data
        self.Word2id = Word2id
        self.Id2word = Id2word
    
    def __getitem__(self, index):
        inp,target,len_content = Generate_tokenizer(self.data.loc[index],self.Word2id) 
        inp_len = len(inp)
        target_len = len(target)
        return {'input':inp,'target':target,'input_len':inp_len,'target_len':target_len,'len_content':len_content}
    
    def __len__(self):
        return len(self.data)
    
    def padding_batch(self,batch):
        input_lens = [d['input_len'] for d in batch]
        target_lens = [d['target_len'] for d in batch]
        len_content = [d['len_content'] for d in batch]
        input_max_len = max(input_lens)
        target_max_len = max(target_lens)
        len_content = max(len_content)    
        
        for d in batch:
            d['input'].extend([self.Word2id['<pad>']] * (input_max_len - d['input_len']))
            d['target'].extend([self.Word2id['<pad>']] * (target_max_len - d['target_len'])) 
        inputs = torch.tensor([d['input'] for d in batch],dtype=torch.long)
        outputs = torch.tensor([d['target'] for d in batch],dtype=torch.long)
        return inputs,outputs,len_content

def re_tokenizer(sentence,Id2word):
    lenth = sentence.size(0)
    str = ''
    for i in range(lenth):
        if i != 0:
            str += Id2word[sentence[i].item()]
    print(str)
        