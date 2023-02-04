import jsonlines
from nltk import word_tokenize
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import AutoTokenizer, AutoModel
import torch
# from get_files_name import listdir
import os
import json
import tokenizer

# 获取同一目录下的所有文件名，返回存储的fileslist
fileslist = []


def listdir(path):
    for file in os.listdir(path):
        # print(file)
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path)
        else:
            filetype = (os.path.splitext(file_path))[1]
            # print(filetype)
            if filetype in ['.c', '.cpp', '.cc', '.txt', '.jsonl']:
                fileslist.append(file_path)
    return fileslist


listdirs = listdir("../data")
f_labels = []
f_codes = []
codes = []
# listdirs = listdirs[0:1]
for dir in listdirs:
    f = open("../data/raw/FFmpeg/train.jsonl", 'r')
    # 从jsonl获取数据,将j标签放入labels
    for item in jsonlines.Reader(f):
        code = item['func']
        f_codes.append(code)
# #加载模型
tokenizer = RobertaTokenizer.from_pretrained("unx/models")

# 获取特殊词列表
count = 0
special_tokens_set = set([])
for f_code in f_codes:
    tokens = word_tokenize(f_code)
    print("这是第%d个代码", count)
    for token in tokens:
        tokens_ids = tokenizer.convert_tokens_to_ids(token)
        if tokens_ids == 3:
            special_tokens_set.add(token)
    count = count + 1
special_tokens_list = list(special_tokens_set)
# 将special_tokens_list写入文本
f = open("../data/special.txt", 'w')
f.writelines(" ".join(str(i) for i in special_tokens_list))
f.close()