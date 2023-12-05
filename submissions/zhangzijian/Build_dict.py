from Data_preprocess import Build_dic_txt
import pandas as pd

if __name__ == '__main__':
    # 读取数据
    raw_train_data = pd.read_json('./data/ccpc_train_v1.0.json',lines=True)
    # 构建字典
    Build_dic_txt(raw_train_data)