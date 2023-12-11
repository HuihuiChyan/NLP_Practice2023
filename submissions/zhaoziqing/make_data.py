import json
from tqdm import tqdm
# 处理数据


def count(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    five, seven = 0, 0
    for i, item in enumerate(tqdm(data)):
        if len(item["content"]) == 23:
            five +=1
        else:
            seven += 1
    print("五言数量：", five)
    print("七言数量：", seven)


def new_data(in_path,out_path):
    with open(in_path, 'r', encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    with open(out_path, 'w', encoding="utf-8") as f:
        lines = []
        for i,item in enumerate(tqdm(data)):
            line = ['标题', '：']+[i for i in item["title"]] + ["***", "格式", ":"]
            if len(item["content"]) == 23:
                line += ["五言"]
            else:
                line += ["七言"]
            line += ["***", "关键词", "："]+[i for i in item["keywords"]] + ["***", "诗歌", "：", "<SEP>"]+[i for i in item["content"]] + ["<SEP>"]
            lines.append(line)
        json.dump(lines, f, ensure_ascii=False)


if __name__ == '__main__':
    #count('./data/ccpc_train_v1.0.json')
    new_data('./data/ccpc_valid_v1.0.json', './data/valid.json')