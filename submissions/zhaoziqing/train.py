from torch import optim
from tqdm import tqdm
from mydataset import *
from eval import eval
from gpt import GPT
device = config.device


def step(model, dataloader, optimizer, criterion, clip=1, print_step=10):
    model.train()
    print_loss = 0
    epoch_loss = 0
    for i, (input, target, poem) in enumerate(tqdm(dataloader)):
        input, target, poem = input.to(device), target.to(device), poem.to(device)
        mask = (target == poem)
        target = target * mask
        output, _ = model(input)
        loss = criterion(output, target.view(-1))
        sum = target.ne(0).long().sum().item()
        loss = loss / sum
        print_loss += loss.item()
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip) #梯度裁剪
        optimizer.step()
        if (i + 1) % print_step == 0:
            avg_loss = print_loss / print_step
            print_loss = 0
            print("Loss: %.4f" % avg_loss)
    return epoch_loss / len(dataloader)


def train(model, dataloader):
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="sum").to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(config.epochs):
        loss = step(model, dataloader, optimizer, criterion, config.clip, config.print_step)
        torch.save(model.state_dict(), 'GPT.pth')
        print("epoch%d Loss: %.4f" % (epoch, loss))
        eval(model)


if __name__ == '__main__':
    print(device)
    model = GPT().to(device)
    # model.load_state_dict(torch.load('GPT.pth'))    # drop=0.1
    data_loader = get_dataloader(config.train_path, config.train_batchsize)
    train(model, data_loader)



