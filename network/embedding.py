import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from nltk import word_tokenize

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")
with open("word.txt", "r") as f:
    special_tokens_list = list(map(str, f.read().split()))
special_tokens_dict = {'additional_special_tokens': special_tokens_list}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
device = torch.device("cpu")
model.to(device)


def mytokenize(text):
    code_tokens = word_tokenize(text)
    tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

    if len(tokens_ids) > 500:
        tokens_ids = tokens_ids[:500]
    context_embeddings = model(torch.tensor(tokens_ids)[None, :].to(device))[0]
    context_embeddings = context_embeddings.detach().numpy()
    context_embeddings = np.pad(context_embeddings, ((0, 0), (0, 500 - len(tokens_ids)), (0, 0)), 'constant',
                                constant_values=-1)
    context_embeddings = np.squeeze(context_embeddings)
    return context_embeddings

# with open("../data/raw/FFmpeg/test.jsonl") as f:
#     a=f.readlines()
#     t=json.loads(a[0])["func"]
# q=mytokenize(t)
# print(q.shape)
# print(q)
