import numpy as np
from tqdm import tqdm
import config
from mydataset import get_dataloader
from gpt import GPT
import torch
device = config.device


def ppl(model, dataloader, loss_fun):
    total_loss = 0
    for i, (input, target, poem) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            input, target, poem = input.to(device), target.to(device), poem.to(device)
            mask = (target == poem)
            target = target * mask
            output, _ = model(input)
            loss = loss_fun(output, target.view(-1))
            sum = target.ne(0).long().sum().item()
            loss = loss / sum
            total_loss += loss.item()

    return total_loss/len(dataloader)

def eval(model):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum").to(device)
    data_loader2 = get_dataloader(config.valid_path, batch_size=32)
    valid_ppl = np.exp(ppl(model, data_loader2, criterion))
    print("valid的困惑度：", valid_ppl)


if __name__ == '__main__':
    print(device)
    model = GPT().to(device)
    model.load_state_dict(torch.load('GPT.pth'))
    model.eval()
    data_loader1 = get_dataloader(config.test_path, batch_size=32)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum").to(device)
    test_ppl = np.exp(ppl(model,data_loader1,criterion))
    data_loader2 = get_dataloader(config.valid_path, batch_size=32)
    valid_ppl = np.exp(ppl(model,data_loader2,criterion))
    print("test的困惑度： %.4f" % test_ppl)
    print("valid的困惑度：%.4f" %  valid_ppl)









