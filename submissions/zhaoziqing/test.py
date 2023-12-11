import torch
import json
import config
from gpt import GPT
"""
测试集前五首诗生成结果
"""
if __name__ == "__main__":
    device = config.device
    model = GPT().to(device)
    model.load_state_dict(torch.load('GPT.pth'))
    model.eval()
    with open(config.test_path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    data = data[:5]
    for item in data:
        index = item.index("<SEP>") + 1
        sentence = item[:index]
        print("".join(sentence))
        result = model.answer(sentence)
        print("结果：", result)

