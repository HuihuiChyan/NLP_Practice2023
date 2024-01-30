import torch


class Config:
    """
        存储所需的参数
    """

    def __init__(self):
        self.train_url = "./data/ccpc_train_v1.0.json"
        self.test_url = "./data/ccpc_test_v1.0.json"
        self.valid_url = "./data/ccpc_valid_v1.0.json"
        self.vocab_url = './data/vocab_lang.txt'
        self.dict_url = "./data/dict.json"
        self.handle_train_url = './data/data_train.json'
        self.handle_test_url = './data/data_test.json'
        self.handle_valid_url = './data/data_valid.json'

        self.max_len = 500
        # self.vocab_size = 7356 + 1
        self.embedding_size = 768  # 词嵌入大小
        self.ffn_size = 2048  # 前馈神经网络参数
        self.num_layers = 6
        self.num_heads = 8
        self.k_size = 64
        self.dropout = 0.1

        self.device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
        self.device_list = torch.cuda.device_count()  # 可用的GPU编号

        self.vocab_list = ["<pad>", "<unk>", "<sep>", "标题", "关键词", "诗歌", "\t", "***", "五言", "七言"]

        self.bacth_size = 64
        self.epoch = 25
        self.learning_rate = 1e-4

        self.max_norm = 1

        self.model_name = "GPT_layers6_8_5.pt"
        self.model_pre = "GPT"

        # 2→cuda：5
        # 4→cuda：5
        # 6→cuda：0
        # 8→cuda：0
        # 10→cuda：0
        # 12→cuda：0
        # last→epoch25,heads8,layers6
