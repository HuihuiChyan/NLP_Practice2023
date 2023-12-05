import torch
from torch import nn
import pandas as pd
import math
from Data_preprocess import *
from PoemModel import *
import time
import pdb
import numpy as np

if __name__ == '__main__':
    # 读取数据,构建词典
    raw_train_data = pd.read_json('./data/ccpc_train_v1.0.json',lines=True)
    dic_pth = "dict.txt"
    Word2id,Id2word = Read_dic(dic_pth)

    # GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型超参数
    vocab_size = len(Word2id)
    d_model = 768
    n_head = 8
    num_layers = 12
    lr = 0.0001
    batch_size = 32

    model = PoemModel(vocab_size,d_model,n_head,num_layers,device,batch_size)
    model.to(device) # 模型 
    loss = torch.nn.CrossEntropyLoss(ignore_index = 8) # 损失函数
    loss.to(device)
    opt = torch.optim.Adam(model.parameters(),lr)# 优化器
    epoch = 30
    
    train_data = MyDataset(raw_train_data,Word2id,Id2word)
    train_data = DataLoader(train_data,batch_size,shuffle=True,drop_last=True,collate_fn=train_data.padding_batch)

    # 训练
    # with open("result.txt","w") as file:
    for i in range(epoch):
        sum_result_loss = 0
        print(f"--------第{i}轮训练开始--------")
        cnt_data = 0
        for batch in train_data:
            opt.zero_grad()
            inp,target = batch
            inp = inp.to(device)
            target = target.to(device) # inp,target batch_size * len_s
            target = target.view(-1)
            # 前向传播
            output = model.forward(inp).to(device) # batch_size * len_s * vocab_size
            output = output.view(-1,vocab_size)
            # 后向传播
            result_loss = loss(output,target) # input (batch_size * len_s,vocab_size) target (1,batch_size * len_s)
            result_loss.backward()
            opt.step()
            
            sum_result_loss += result_loss
            # print(f"batch{cnt_data}的损失值为{result_loss.item()}")
            cnt_data += 1
        print(f"第{i}轮的平均损失为{sum_result_loss.item()/cnt_data}")

    # 保存模型参数
    torch.save(model.state_dict(),'./model_state.pth')
