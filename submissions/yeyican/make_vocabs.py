import json

from config import Config
from train import write_traindata_tofile


def read_from_json(url):
    """
    从文件中按行解析json
    :param url: 文件路径
    :return: json对象列表
    """
    results = []
    with open(url, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            line_json = json.loads(line.rstrip('\n'))
            results.append(line_json)
            line = f.readline()
    f.close()
    return results


def make_dict(url):
    """
    从文件中读取数据，生成char2index、index2char和字符列表
    :param url: 文件路径
    :return: char2index，index2char，字符列表
    """
    vocab_char_to_index = {}
    vocab_index_to_char = {}
    voc_list = []
    with open(url, 'r', encoding='utf-8') as f:
        line = f.readline()
        num = 0
        while line:
            line_str = line.rstrip('\n')
            if line_str == '':
                line = f.readline()
                continue
            vocab_char_to_index[line_str] = num
            vocab_index_to_char[num] = line_str
            voc_list.append(line_str)
            num += 1
            line = f.readline()
    f.close()
    return vocab_char_to_index, vocab_index_to_char, voc_list


def make_vocab_from_json(url_list):
    """
    解析诗歌文件，生成对应字符集合
    :param url_list: 文件路径
    :return: 字符集合
    """
    vocab_set = set()
    key_list = ['content', 'title', 'keywords']  # 数据中需要分词的key
    for url in url_list:
        with open(url, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                line_str = line.rstrip('\n')
                obj_json = json.loads(line_str)
                for key in key_list:
                    chars = obj_json[key]
                    for char in chars:
                        vocab_set.add(char)
                line = f.readline()
        f.close()
    return vocab_set


def write_vocab(set, url, vocab_others):
    """
    将字符集写回指定文件
    :param set: 字符集合
    :param url: 文件路径
    :param vocab_others：词表中特殊字符
    :return: 无返回值
    """
    with open(url, 'w', encoding='utf-8') as f:
        for str in vocab_others:
            print(str, file=f)
        for char in set:
            print(char, file=f)
    print(f"已将分词结果写入{url}")


def write_dicts(vocab_url, dict_url):
    """
        保存对应关系

        特殊字符：<pad> <unk> <sep> 标题 关键词 诗歌 *** : '    '
    """

    """
        获取字符到index、index到字符的映射，并存储至指定路径
    """
    vocab_char_to_index, vocab_index_to_char, vocab_list = make_dict(vocab_url)
    # print(vocab_char_to_index)
    # print(vocab_index_to_char)
    # print(vocab_list)

    # s = set()
    # for num in vocab_index_to_char:
    #     char = vocab_index_to_char[num]
    #     if num != vocab_char_to_index[char]:
    #         print("?????")
    #         print(char)
    #     if char != vocab_index_to_char[num]:
    #         print("!!!!!!")
    #     if s.__contains__(char):
    #         print(s)
    #     else:
    #         s.add(char)
    print(len(vocab_char_to_index), len(vocab_index_to_char), len(vocab_list))

    obj = {
        "char2index": vocab_char_to_index,
        "index2char": vocab_list
    }
    with open(dict_url, 'w', encoding='utf-8') as file:
        json.dump(obj=obj, fp=file, ensure_ascii=False)
    file.close()


if __name__ == "__main__":
    """
        1.从文件中读取数据，得出所有的字符
        2.将字符存储在文件中（可以只执行一次）
        3.读取字符集，构建词表
        4.将词表写入文件中
    """
    config = Config()
    url_list = [config.train_url, config.valid_url, config.test_url]
    vocab_set = make_vocab_from_json(url_list)
    write_vocab(vocab_set, config.vocab_url, vocab_others=config.vocab_list)
    write_dicts(vocab_url=config.vocab_url, dict_url=config.dict_url)
    write_traindata_tofile(config.train_url, config.handle_train_url)
    write_traindata_tofile(config.test_url, config.handle_test_url)
    write_traindata_tofile(config.valid_url, config.handle_valid_url)
