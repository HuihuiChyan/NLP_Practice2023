import json
import torch
import torch.utils.data as Data
from torch import nn, optim
import numpy as np
import time
from tqdm import tqdm
from gpt_model import *

def make_data(datas):
    train_datas = []
    for data in datas:
        train_data = [i if i!='\t' else "<sep>" for i in data]+['<sep>']
        train_datas.append(train_data)

    return train_datas


class MyDataSet(Data.Dataset):
    def __init__(self,datas):
        self.datas = datas

    def __getitem__(self, item):
        data = self.datas[item]
        #print(data)
        #print("".join([id2word[i] for i in data]))
        idx = data.index(5) + 2
        decoder_input = data[:-1]
        decoder_output = data[1:]
        poem_output = data[1:]
        for i in range(idx):
            poem_output[i] = 0

        decoder_input_len = len(decoder_input)
        decoder_output_len = len(decoder_output)

        return {"decoder_input":decoder_input,"decoder_input_len":decoder_input_len,
                "decoder_output":decoder_output,"decoder_output_len":decoder_output_len,"poem_output":poem_output}

    def __len__(self):
        return len(self.datas)

    def padding_batch(self,batch):
        decoder_input_lens = [d["decoder_input_len"] for d in batch]
        decoder_output_lens = [d["decoder_output_len"] for d in batch]

        decoder_input_maxlen = max(decoder_input_lens)
        decoder_output_maxlen = max(decoder_output_lens)


        for d in batch:
            d["decoder_input"].extend([word2id["<pad>"]]*(decoder_input_maxlen-d["decoder_input_len"]))
            d["decoder_output"].extend([word2id["<pad>"]]*(decoder_output_maxlen-d["decoder_output_len"]))
            d["poem_output"].extend([word2id["<pad>"]]*(decoder_output_maxlen-d["decoder_output_len"]))
        decoder_inputs = torch.tensor([d["decoder_input"] for d in batch],dtype=torch.long)
        decoder_outputs = torch.tensor([d["decoder_output"] for d in batch],dtype=torch.long)
        poem_outputs = torch.tensor([d["poem_output"] for d in batch],dtype=torch.long)

        return decoder_inputs,decoder_outputs,poem_outputs
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train_step(model,data_loader,optimizer,criterion,clip=1,print_every=None):
    model.train()

    if print_every == 0:
        print_every = 1

    print_loss_total = 0  # 每次打印都重置

    epoch_loss = 0

    for i, (dec_inputs, dec_outputs, poem_outputs) in enumerate(tqdm(data_loader)):
        '''
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        '''
        optimizer.zero_grad()
        dec_inputs, dec_outputs, poem_outputs =dec_inputs.to(device), dec_outputs.to(device), poem_outputs.to(device)
        mask = poem_outputs == dec_outputs
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, dec_self_attns = model(dec_inputs)
        dec_outputs *= mask
        #print(outputs.shape, dec_outputs.shape)

        loss = criterion(outputs, dec_outputs.view(-1))
        num = dec_outputs.ne(0).long().sum().item()
        loss = loss / num
        print_loss_total += loss.item()
        epoch_loss += loss.item()
        loss.backward()


        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        if print_every and (i + 1) % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('\tCurrent Loss: %.4f' % print_loss_avg)

    return epoch_loss / len(data_loader)

def train(model,data_loader):
    criterion = nn.CrossEntropyLoss(ignore_index=0,  reduction="sum").to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train_step(model, data_loader, optimizer, criterion, CLIP, print_every=100)
        end_time = time.time()

        torch.save(model.state_dict(), 'GPT2.pt')

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')



def print_num_parameters(model):
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(

        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

if __name__ == '__main__':
    with open('dataset.json', 'r', encoding='utf-8') as f:
        datas = json.load(f)

    train_data = make_data(datas)

    train_num_data = [[word2id[word] for word in line] for line in train_data]

    batch_size = 8
    epochs = 30
    dataset = MyDataSet(train_num_data)
    data_loader = Data.DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.padding_batch)

    model = GPT().to(device)

    model_size = sum([i.numel() for i in model.parameters()])
    print(f"model_size:{model_size / 1000 / 1000} M")
    # model.load_state_dict(torch.load('GPT2.pt'))

    train(model,data_loader)
