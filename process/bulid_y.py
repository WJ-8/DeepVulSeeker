import os
import json
import numpy as np
num_test = []
num_train = []
num_val = []
with open(r'F:\zsw\test2\test.jsonl','r',encoding='utf8')as f:
    c = f.readlines()
    for i in c:
        text = json.loads(i)["target"]
        # print(text)
        num_test.append(int(text))
num1 = np.array(num_test)
print(num1.shape)
np.save('F:\\zsw\\test2\\npy\\test_y.npy',num1)
f.close()
# with open(r'E:\magic-cb\data\raw\FFmpeg\train.jsonl','r',encoding='utf8')as f:
#     c = f.readlines()
#     for i in c:
#         text = json.loads(i)["target"]
#         # print(text)
#         num_train.append(int(text))
# num2 = np.array(num_train)
# print(num2.shape)
# np.save('F:\\zsw\\no_self_connect\\train_y.npy',num2)
# f.close()
# with open(r'E:\magic-cb\data\raw\FFmpeg\valid.jsonl','r',encoding='utf8')as f:
#     c = f.readlines()
#     for i in c:
#         text = json.loads(i)["target"]
#         # print(text)
#         num_val.append(int(text))
# num3 = np.array(num_val)
# print(num3.shape)
# np.save('F:\\zsw\\no_self_connect\\val_y.npy',num3)
# f.close()