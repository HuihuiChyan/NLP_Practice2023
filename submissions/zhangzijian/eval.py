import torch
from torch import nn
import pandas as pd
import math
from torch.utils.data import DataLoader,Dataset
from PoemModel import PoemModel
from Data_preprocess import * 
import pdb        
if __name__ == '__main__':
    # 构建字典
    dict_pth = "dict.txt"
    Word2id,Id2word = Read_dic(dict_pth)

    # GPU设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型超参数
    vocab_size = len(Word2id)
    d_model = 768
    n_head = 8
    num_layers = 12
    batch_size = 1
    
    model = PoemModel(vocab_size,d_model,n_head,num_layers,device,batch_size)
    model.to(device)
    model.load_state_dict(torch.load('model_state.pth')) # 模型
    loss = torch.nn.CrossEntropyLoss(ignore_index = 8,reduction = 'mean') # 损失函数
    loss.to(device)
    
    # # 读取数据
    raw_test_data = pd.read_json('./data/ccpc_valid_v1.0.json',lines=True)
    test_data = Mytest_Dataset(raw_test_data,Word2id,Id2word)
    test_data = DataLoader(test_data,batch_size,shuffle=True,drop_last=True,collate_fn=test_data.padding_batch)
    
    # 生成,并输出PPL
    sum_Loss = 0
    lenth = 0
    sf = nn.Softmax(dim = -1)
    with torch.no_grad():
        for batch in test_data:
            inp,target,len_content = batch
            inp = inp.to(device)
            target = target.to(device)
            target = target.view(-1)
            
            re_tokenizer(inp.view(-1),Id2word)
            
            output_sen = ''
            output_logit = torch.zeros([len_content,vocab_size],dtype = torch.float32,device = device)
            for i in range(len_content):
                result = model.forward(inp) # input:(1 , len_s)  result:(1 , len_s , vocab_size)
                # 获取下一个token
                next_token = result[0,result.size(1) - 1,:]
                output_logit[i,:] = next_token
                # greedy decoder
                next_logit = sf(next_token) # 对输出进行softmax
                token = torch.argmax(next_logit,dim = -1) # 选择词表上最大概率的词
                output_sen += Id2word[token.item()]
                inp = torch.cat((inp,token.unsqueeze(0).unsqueeze(0)),dim = 1)
            poem_loss = loss(output_logit,target)
            print(f"{output_sen},ppl:{torch.exp(poem_loss)}\n")
            sum_Loss += poem_loss
            lenth += 1
    mean_loss = sum_Loss / lenth
    print(f"在整个数据集上的平均PPL为{torch.exp(mean_loss)}")
                

            
            
            
            
    
    
