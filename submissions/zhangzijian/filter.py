from Data_preprocess import Read_dic
import pdb
if __name__ == '__main__':
    # 字典文件存在问题，读入两个字典长度不等，运行此程序能够找到发生错误的大致位置 6970
    dic_pth = "dict.txt"
    Word2id,Id2word = Read_dic(dic_pth)
    lenth = len(Id2word)
    for i in range(lenth):
        word = Id2word[i]
        if Word2id[word] != i:
            print(f"{i}\n")
    print(f"{len(Word2id)},{len(Id2word)}")