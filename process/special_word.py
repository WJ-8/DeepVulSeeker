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
    f = open('F:\\zsw\\test2\\test.jsonl', 'r')
    # f = open(dir, 'r')
    # 从jsonl获取数据,将j标签放入labels
    for item in jsonlines.Reader(f):
        # lable = item['target']
        # f_labels.append(lable)
        code = item['func']
        f_codes.append(code)
# #加载模型
tokenizer = RobertaTokenizer.from_pretrained("F:\\zsw\\unx\\models")
# model = RobertaModel.from_pretrained("microsoft/codebert-base")

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
f = open("F:\\zsw\\pythonProject\\special_tokens_list_test2.txt", 'w')
f.writelines(" ".join(str(i) for i in special_tokens_list))
f.close()

# # 将special_tokens_list加入到tokenizer中
# print(special_tokens_list)
# print("添加前长度",len(tokenizer))
# special_tokens_dict = {'additional_special_tokens':special_tokens_list}
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# # model.resize_token_embeddings(len(tokenizer))
# print("添加后长度",len(tokenizer))
# print(tokenizer.convert_tokens_to_ids('subframe_count_exact'))
#
# # 获取源代码的embedding
# str = '''static int frame_count_exact ( FlacEncodeContext * s , FlacSubframe * sub ,
#  int pred_order )
#  {
#  int p , porder , psize ;
#  int i , part_end ;
#  int count = 0 ;
#  count += 8 ;
#  if ( sub -> type == FLAC_SUBFRAME_CONSTANT ) {
#  count += sub -> obits ;
#  } else if ( sub -> type == FLAC_SUBFRAME_VERBATIM ) {
#  count += s -> frame . blocksize * sub -> obits ;
#  } else {
#  count += pred_order * sub -> obits ;
#  if ( sub -> type == FLAC_SUBFRAME_LPC )
#  count += 4 + 5 + pred_order * s -> options . lpc_coeff_precision ;
#  count += 2 ;
#  porder = sub -> rc . porder ;
#  psize = s -> frame . blocksize >> porder ;
#  count += 4 ;
#  i = pred_order ;
#  part_end = psize ;
#  for ( p = 0 ; p < 1 << porder ; p ++ ) {
#  int k = sub -> rc . params [ p ] ;
#  count += 4 ;
#  count += rice_count_exact ( & sub -> residual [ i ] , part_end - i , k ) ;
#  i = part_end ;
#  part_end = FFMIN ( s -> frame . blocksize , part_end + psize ) ;
#  }
#  }
#  return count ;
#  }'''
# code_tokens= word_tokenize(str)
# tokens=[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
# tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
# print(len(tokens_ids))
# if(len(tokens_ids)>500):
#     tokens_ids = tokens_ids[0:500]
# else:
#     for i in range(len(tokens_ids),500):
#         tokens_ids.append(0)
# print(tokens_ids)
# context_embeddings=model(torch.tensor(tokens_ids)[None,:])[0]
# print(context_embeddings.shape)
