import json

def get_dict(path):
    vocab_dict, vocab_list = {}, []
    with open(path, encoding="utf-8", mode='r') as f:
        for i, l in enumerate(f.readlines()):
            vocab_dict[l.rstrip("\n")] = i
            vocab_list.append(l.rstrip("\n"))
    return vocab_dict, vocab_list

def read_data(path):
    with open(path, encoding="utf-8", mode='r') as f:
        data = [json.loads(l) for l in f.readlines()]
    return data

vocab_dict, vocab_list = get_dict("./vocab.txt")
data = read_data("../CCPC/ccpc_test_v1.0.json")

'''with open('dict_datas.json', 'w', encoding='utf-8') as f:
    json.dump({'word2id':vocab_dict, 'id2word':vocab_list}, f, ensure_ascii=False)'''

with open('dataset_test.json', 'w', encoding='utf-8') as f:
    lines = []
    for text_data in data:
        line = ["标题", "："] + [w for w in text_data["title"]] + ["***", "关键词", "："]
        if len(text_data["content"]) == 23:
            line += ["五言", " "] + [w for w in text_data["keywords"]] + ["***", "诗歌", "：", "\t"]
        else:
            line += ["七言", " "] + [w for w in text_data["keywords"]] + ["***", "诗歌", "：", "\t"]
        line += [w for w in text_data["content"]]
        lines.append(line)
    json.dump(lines, f, ensure_ascii=False)
