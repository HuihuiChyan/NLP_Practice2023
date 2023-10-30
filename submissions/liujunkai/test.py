import torch

from gpt_model import GPT

if __name__ == '__main__':
    device = torch.device('cuda')
    model = GPT().to(device)
    model.load_state_dict(torch.load('GPT2_small.pt'))

    model.eval()
    #初始输入是空，每次加上后面的对话信息
    sentence = []
    while True:
        temp_sentence = input("标题:")
        sentence += (["标题", "："] + [w for w in temp_sentence] + ["***"])
        temp_sentence = input("关键词:")
        sentence += (["关键词", "：", "五言", " "] + [w for w in temp_sentence] + ["***", "诗歌", "：", "<sep>"])
        if len(sentence) > 200:
            #由于该模型输入最大长度为300，避免长度超出限制长度过长需要进行裁剪
            t_index = sentence.find('<sep>')
            sentence = sentence[t_index + 1:]
        print("机器人:", model.answer(sentence))
