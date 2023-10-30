import torch
import tqdm
import json
import math

from train import *
from gpt_model import GPT
from torch.nn import functional as F

def get_score(model,data_loader,criterion,print_every=None):
    epoch_loss = 0
    for i, (dec_inputs, dec_outputs, poem_outputs) in enumerate(tqdm(data_loader)):
        '''
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        '''
        with torch.no_grad():
            dec_inputs, dec_outputs, poem_outputs =dec_inputs.to(device), dec_outputs.to(device), poem_outputs.to(device)
            mask = poem_outputs == dec_outputs
            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            outputs, dec_self_attns = model(dec_inputs)
            dec_outputs *= mask
            #print(outputs.shape, dec_outputs.shape)

            loss = criterion(outputs, dec_outputs.view(-1))
            num = dec_outputs.ne(0).long().sum().item()
            loss = loss / num
            epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

def make_data(datas):
    train_datas =[]
    for data in datas:
        train_data = [i if i!='\t' else "<sep>" for i in data]+['<sep>']
        train_datas.append(train_data)

    return train_datas

if __name__ == '__main__':
    device = torch.device('cuda')
    model = GPT().to(device)
    model.load_state_dict(torch.load('GPT2_small.pt'))

    dict_datas = json.load(open('dict_datas.json', 'r'))
    word2id, id2word = dict_datas['word2id'], dict_datas['id2word']
    id2word.append("\n")
    word2id["\n"] = len(id2word)-1

    with open('dataset_test.json', 'r', encoding='utf-8') as f:
        datas = json.load(f)
    test_data = make_data(datas)
    test_num_data = [[word2id[word] for word in line] for line in test_data]
    dataset = MyDataSet(test_num_data)
    data_loader = Data.DataLoader(dataset, batch_size=8, collate_fn=dataset.padding_batch)
    criterion = nn.CrossEntropyLoss(ignore_index=0,  reduction="sum").to(device)
    perplexity = get_score(model,data_loader,criterion,print_every=None)
    print(f"perplexity: {perplexity}")

