from tqdm import tqdm
import pandas as pd
import json

# with open('dataset/poetryTang/poetryTang.txt', 'r', encoding='utf-8') as handler:
#     lines = handler.read().split('\n')
#
#     data = list()
#     for line in tqdm(lines):
#         sp = line.split('::')
#         if len(sp) != 3:
#             print("Error: ", sp)
#             continue
#         data.append(sp)
# train = pd.DataFrame(data)
# train.columns = ['title', 'author', 'content']
# train['keywords']=['']*len(train)
# train['dynasty']=['Tang']*len(train)
# train.to_csv('dataset/poetryTang/poetryTang.csv',columns=['title', 'author', 'content','keywords','dynasty'],
#             sep='\t',
#             index=False)

json_list = []
with open('./dataset/CCPC/ccpc_train_v1.0.json','r',encoding='utf-8')as fp:
    for line in fp.readlines():
        json_list.append(json.loads(line))

data = [[d['dynasty'],d['author'],d['content'].replace('|',',')+'。',d['title'],d['keywords']] for d in json_list]
train = pd.DataFrame(data,columns = ['dynasty','author','content','title','keywords'])
train.head()
train.to_csv('dataset/CCPC/CCPC.csv',columns=['title', 'author', 'content','keywords','dynasty'],
            sep='\t',
            index=False)


import matplotlib.pyplot as plt

texts = train['content'].tolist()
dict = {}
for i in range(len(texts)):
    l = len(texts[i])
    if l not in dict:
        dict[l] = 1
    else:
        dict[l] = dict[l] + 1

d = sorted(dict.items(), key=lambda k: k[0])

x = []
y = []
for i in range(len(d)):
    x.append(d[i][0])
    y.append(d[i][1])
plt.bar(x, y)
plt.savefig('CCPC.png')
plt.show()
