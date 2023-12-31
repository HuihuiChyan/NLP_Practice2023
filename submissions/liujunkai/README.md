# GPT架构的诗歌生成模型

## 引用代码

[主体部分](https://blog.csdn.net/weixin_44599230/article/details/)

[mask修改](https://github.com/liucongg/GPT2-NewsTitle/blob/main/model.py)

## 运行说明

make_data.py用于构造数据集；train.py训练模型；test.py测试模型；eval.py计算困惑度。路径需要自行修改。

## 测试结果

|  dataset |  eval | test  |
|---|---|---|
| perplexity  | 7.51  | 7.50 |

## 结果样例
```
标题:咏史二十二首 其十六 隋文帝
关键词:五言 贻谋 学术 侯 失
机器人: 正道丘西道益头|偶然学术不为修|养高处处天临广|正是当年学术侯

标题:漫兴九首 其二
关键词:七言 土 识 汹汹 战场
机器人: 汹汹陂头汹汹隈|民间一饱不为灾|早知身世几时合|犹送汹汹上沛来

标题:别离情所钟十二章章四句送定叟弟之官严陵 其十二
关键词:五言 短 吹我 朔风 送子
机器人: 眼昏犹觉短纵横|吹我哭声与别情|我愧短蓑芦雁过|与君吹我凯风横

标题:席上分得时字送豫斋二首 其一
关键词:五言 见月 客中 泽国 登楼
机器人: 泽国登楼泽国心|登楼见月免登楼|不知今夜征轮息|只有飞鸢不肯愁

标题:羊城八景 其八 扶桑浴日
关键词:五言 远空 东溟 金轮 洗出
机器人: 东溟一望远空流|群从金沟洗出浮|不遣沉芒逐伴去|汉兵几力避渔舟
```

## 缺陷

超参数部分在gpt_model.py中。当前模型无法很好地根据提示生成五言或七言，不知是训练问题还是模型大小问题。

![效果展示](1.png)
