import json

import torch
from gpt import GPT
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from config import Config
from train import *


def make_data(datas):
    use_datas = []
    for data in datas:
        use_data = [i if i != '\t' else "<sep>" for i in data] + ['<sep>']
        use_datas.append(use_data)

    return use_datas


def get_score(model, data_loader, cross):
    epoch_loss = 0
    for (inputs, outputs, poems) in data_loader:
        with torch.no_grad():
            inputs = inputs.to(device)
            outputs = outputs.to(device)
            poems = poems.to(device)
            mask = (poems == outputs)
            output, attns = model(inputs)
            outputs *= mask
            loss = cross(output, outputs.view(-1))
            num = outputs.ne(0).long().sum().item()
            loss = loss / num
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


def test(dataset_url, test_type):
    print(f"{test_type}：")
    with open(dataset_url, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    formdata = make_data(datas)
    formdata_num = [[char2index[word] for word in line] for line in formdata]
    dataset = PoemDataSet(formdata_num)
    data_loader = DataLoader(dataset, batch_size=64, collate_fn=dataset.collate_fn)
    cross = nn.CrossEntropyLoss(ignore_index=0, reduction="sum").to(device)
    perplexity = get_score(model, data_loader, cross)
    print(f"perplexity: {perplexity}")
    return perplexity


if __name__ == '__main__':
    config = Config()
    device = config.device
    model = GPT().to(device)

    model.load_state_dict(torch.load(
        "GPT_last.pt"
    ))
    dict_datas = json.load(open('./data/dict.json', 'r'))
    char2index, index2char = dict_datas['char2index'], dict_datas['index2char']
    index2char.append("\n")
    char2index["\n"] = len(index2char) - 1

    print("-------------------")
    test(config.handle_test_url, "测试集")
    print("-------------------")
    test(config.handle_valid_url, "验证集")
    print("-------------------")
    print("人工输入部分：")
    model.eval()

    titles = ["咏史二十二首 其十六 隋文帝",
            "别离情所钟十二章章四句送定叟弟之官严陵 其十二",
            "席上分得时字送豫斋二首 其一",
            "芙蓉",
            "羊城八景 其八 扶桑浴日"]
    keys = ["五言 贻谋 学术 侯 失",
            "五言 短 吹我 朔风 送子",
            "五言 见月 客中 泽国 登楼",
            "五言 云锦 南州 青松 芙蓉",
            "五言 远空 东溟 金轮 洗出"]
    # keys = ["贻谋 学术 侯 失",
    #         "短 吹我 朔风 送子",
    #         "见月 客中 泽国 登楼",
    #         "云锦 南州 青松 芙蓉",
    #         "远空 东溟 金轮 洗出"]
    # nums = ['五言','五言','五言','五言','五言']

    for index in range(5):
        sentence = []
        temp_sentence = titles[index]
        sentence += (["标题", "："] + [w for w in temp_sentence] + ["***"])
        temp_sentence = keys[index]
        sentence += (["关键词", "："] + [w for w in temp_sentence] + ["***", "诗歌", "：", "<sep>"])
        # sentence += (["格式", "：", nums[index]] + ["***", "关键词", "："] + [w for w in temp_sentence] + ["***", "诗歌", "：", "<sep>"])
        if len(sentence) > 200:
            # 由于该模型输入最大长度为300，避免长度超出限制长度过长需要进行裁剪
            t_index = sentence.find('<sep>')
            sentence = sentence[t_index + 1:]
        print(f"测试集第{index + 1}条的生成结果为:{model.answer(sentence)}")

    # test_loss = []
    # valid_loss = []
    # for epoch in range(config.epoch):
    #     print(f"./model/{config.model_pre}_{config.num_layers}_{config.num_heads}_{epoch+1}.pt:")
    #     model.load_state_dict(torch.load(
    #         f"./model/{config.model_pre}_{config.num_layers}_{config.num_heads}_{epoch+1}.pt"
    #     ))
    #     dict_datas = json.load(open('./data/dict.json', 'r'))
    #     char2index, index2char = dict_datas['char2index'], dict_datas['index2char']
    #     index2char.append("\n")
    #     char2index["\n"] = len(index2char) - 1

    #     print("-------------------")
    #     t_p = test(config.handle_test_url, "测试集")
    #     test_loss.append(t_p)
    #     print("-------------------")
    #     v_p = test(config.handle_valid_url, "验证集")
    #     valid_loss.append(v_p)
    #     print("-------------------")
    #     print("人工输入部分：")
    #     model.eval()

    #     titles = ["咏史二十二首 其十六 隋文帝",
    #             "别离情所钟十二章章四句送定叟弟之官严陵 其十二",
    #             "席上分得时字送豫斋二首 其一",
    #             "芙蓉",
    #             "羊城八景 其八 扶桑浴日"]
    #     keys = ["五言 贻谋 学术 侯 失",
    #             "五言 短 吹我 朔风 送子",
    #             "五言 见月 客中 泽国 登楼",
    #             "五言 云锦 南州 青松 芙蓉",
    #             "五言 远空 东溟 金轮 洗出"]
    #     # keys = ["贻谋 学术 侯 失",
    #     #         "短 吹我 朔风 送子",
    #     #         "见月 客中 泽国 登楼",
    #     #         "云锦 南州 青松 芙蓉",
    #     #         "远空 东溟 金轮 洗出"]
    #     # nums = ['五言','五言','五言','五言','五言']

    #     for index in range(5):
    #         sentence = []
    #         temp_sentence = titles[index]
    #         sentence += (["标题", "："] + [w for w in temp_sentence] + ["***"])
    #         temp_sentence = keys[index]
    #         sentence += (["关键词", "："] + [w for w in temp_sentence] + ["***", "诗歌", "：", "<sep>"])
    #         # sentence += (["格式", "：", nums[index]] + ["***", "关键词", "："] + [w for w in temp_sentence] + ["***", "诗歌", "：", "<sep>"])
    #         if len(sentence) > 200:
    #             # 由于该模型输入最大长度为300，避免长度超出限制长度过长需要进行裁剪
    #             t_index = sentence.find('<sep>')
    #             sentence = sentence[t_index + 1:]
    #         print(f"测试集第{index + 1}条的生成结果为:{model.answer(sentence)}")

    # with open(f"./loss/test_6_8.json" ,"w",encoding='utf-8') as file:
    #     json.dump(test_loss,file,ensure_ascii=False)
    # file.close()
    # with open(f"./loss/valid_6_8.json" ,"w",encoding='utf-8') as file:
    #     json.dump(valid_loss,file,ensure_ascii=False)
    # file.close()

    while True:
        sentence = []
        temp_sentence = input("标题:")
        sentence += (["标题", "："] + [w for w in temp_sentence] + ["***"])
        # temp_sentence = input("格式:")
        # sentence += (["格式", "：", temp_sentence] + ["***"])
        temp_sentence = input("关键词:")
        sentence += (["关键词", "："] + [w for w in temp_sentence] + ["***", "诗歌", "：", "<sep>"])
        # sentence += (["关键词", "："] + [w for w in temp_sentence] + ["***", "诗歌", "：", "<sep>"])
        if len(sentence) > 200:
            # 由于该模型输入最大长度为300，避免长度超出限制长度过长需要进行裁剪
            t_index = sentence.find('<sep>')
            sentence = sentence[t_index + 1:]
        print("生成结果:", model.answer(sentence))
