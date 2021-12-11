# Chinese-Poem-Generate-Based-on-GPT2

# 基于GPT-2的诗词生成器 #
## 包含功能：唐诗生成、宋诗生成、宋词生成、绝句生成（关键词） ##
## 目录结构 ##

- dataset：使用的数据集

	CCPC: 绝句10w首

	poetryTang: 唐诗5w首

	poetrySong: 宋诗26w首，默认训练数据

	songci: 宋词，2w首

- model_CCPC:关键词写诗模型

- model_ci:宋词生成模型

- model_song:宋诗生成模型

- model_tang:唐诗生成模型

- LoadData.py：数据预处理

- MyModel.py：修改后的GPT2模型

- RoleData.py：自定义数据类型

- Train.py：模型训练

- Generate.py：诗词生成

## 1. 开始训练 ##

python Train.py --mode tang/song/ci/CCPC

## 2. 生成诗词 ##

python Generate.py --mode tang/song/ci/CCPC

python Generate.py --mode CCPC
运行结果：

输入的字符为:家国 秋水 悲

生成的第1个诗为：家国千秋月正清,不因相望有情思,谁能为问当年事,一日无愁忆旧楼。

生成的第2个诗为：十年无处在江湖,秋水如霜不尽多,何限一番愁国事,更将秋水作悲舟。

生成的第3个诗为：家国秋风动不知,一般愁思寄何须,自缘未觉秋江水,为有离人亦断肠。

生成的第4个诗为：秋水连城去,愁云欲动悲,一声千里夜,只在此中归。

生成的第5个诗为：秋水生平国,荒池不敢期,何堪为客恨,相忆楚山涯。

python Generate.py --mode song
运行结果：

输入的字符为:春风

生成的第1个诗为：春风满目柳阴晴，小雨微茫绿欲明。无赖故园春未了，更将闲卧看花行。

生成的第2个诗为：春风来往还，桃李未成雪。天寒雨如何，山色无人语。

生成的第3个诗为：春风吹雨湿，野店度行村。水面山连树，天开鸟过门。竹深苔藓合，花晚石桥存。客子何时去，归来未到家。

生成的第4个诗为：春风吹雨过西窗，不为山河一径青。自喜清光无俗物，只应幽梦到柴门。

生成的第5个诗为：春风吹我归，花影上天涯。

## 模型文件 ##

模型已上传至百度云盘，下载后替换对应文件夹即可。

绝句:

链接：https://pan.baidu.com/s/1m-PAHrnAjeQUaJk7ZxT8hg 
提取码：vwls

宋词:

链接：https://pan.baidu.com/s/1P-QACK1NU4S8Lvd_ex0FBA 
提取码：eqb9

宋诗:

链接：https://pan.baidu.com/s/1C16jusNfkihds3AsdbRz9A 
提取码：5ylb

唐诗:

链接：https://pan.baidu.com/s/1fx3tRl3w5nP8YkpKeLIT_A 
提取码：q2cq

## 参考链接 ##

https://github.com/liucongg/GPT2-NewsTitle

https://github.com/chinese-poetry/chinese-poetry

https://github.com/THUNLP-AIPoet/
