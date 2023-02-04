import os
import time

import numpy as np
import torch
from unixcoder import UniXcoder
import json
from transformers import AutoTokenizer, AutoModel


model = UniXcoder("F:\\zsw\\unx\\models")
device = torch.device("cpu")
model.to(device)

# my_dic = {}
# with open("E:\\zsw\\unx\\models\\vocab.json", "r+", encoding="utf8") as m:
#     data = json.loads(m.read().replace("Ġ", " "))
#     for k, v in data.items():
#         my_dic[v] = k


# Encode maximum function
path = 'F:\\zsw\\test2\\c'
path_list = os.listdir(path)
num = len(os.listdir(path))
times = 0
for i in range(0,num):
    times += 1
        # text = json.loads(i)["func"].replace("\n", "")
    with open(path+'\\'+str(path_list[i]),'r',encoding='utf8')as f:
        text = f.read().replace("\n",'')
        func = text
        # print(func)
        # time.sleep(10)
        tokens_ids = model.tokenize([func], max_length=512, mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(device)
        tokens_embeddings, max_func_embedding = model(source_ids)
        max_func_embedding = np.expand_dims(max_func_embedding.detach().numpy(), 0)
        # print(tokens_ids)
        # print("次数："+str(i))
        if times == 1:
            embed = max_func_embedding
        elif times > 1:
            embed = np.concatenate((embed, max_func_embedding), axis=0)
        print(embed.shape)


np.save('F:\\zsw\\test2\\npy\\test_emb.npy', embed)
# n = np.load('E:\\magic-cb\\data\\embedding\\val_emb.npy')
# print(n[0])
# func = "def f(a,b): if a>b: return a else return b"
# tokens_ids = model.tokenize([func],max_length=512,mode="<encoder-only>")
# source_ids = torch.tensor(tokens_ids).to(device)
# tokens_embeddings,max_func_embedding = model(source_ids)

# Encode minimum function
# func = "def f(a,b): if a<b: return a else return b"
# tokens_ids = model.tokenize([func], max_length=512, mode="<encoder-only>")
# source_ids = torch.tensor(tokens_ids).to(device)
# tokens_embeddings, min_func_embedding = model(source_ids)

# Encode NL
# nl = "return maximum value"
# tokens_ids = model.tokenize([nl], max_length=512, mode="<encoder-only>")
# source_ids = torch.tensor(tokens_ids).to(device)
# tokens_embeddings, nl_embedding = model(source_ids)

# print(max_func_embedding.shape)
# print(max_func_embedding)
