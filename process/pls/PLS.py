import os
import time

import numpy as np
import torch
from unixcoder import UniXcoder
import json
from transformers import AutoTokenizer, AutoModel


model = UniXcoder("../unx/models")
device = torch.device("cpu")
model.to(device)


# Encode maximum function
path = "../../data/c"
path_list = os.listdir(path)
num = len(os.listdir(path))
times = 0
for i in range(0,num):
    times += 1
        # text = json.loads(i)["func"].replace("\n", "")
    with open(path+'\\'+str(path_list[i]),'r',encoding='utf8')as f:
        text = f.read().replace("\n",'')
        func = text
        tokens_ids = model.tokenize([func], max_length=512, mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(device)
        tokens_embeddings, max_func_embedding = model(source_ids)
        max_func_embedding = np.expand_dims(max_func_embedding.detach().numpy(), 0)
        if times == 1:
            embed = max_func_embedding
        elif times > 1:
            embed = np.concatenate((embed, max_func_embedding), axis=0)
        print(embed.shape)


np.save('../../data/embedding/datatest_emb.npy', embed)