import json

cnt = 1
with open("F:\\zsw\\test2\\test.jsonl",'r')as f:
    c = f.readlines()

#对多个jsonl文件合并
# with open("E:\\magic-cb\\data\\raw\\qemu\\train.jsonl",'r')as k:
#     c+=k.readlines()
# with open("E:\\magic-cb\\data\\raw\\qemu\\valid.jsonl", 'r') as m:
#     c += m.readlines()
#     with open("F:\\zsw\\ffqejson\\qemu.jsonl",'w')as m:
#         for i in c:
#             cnt+=1
#             m.write(i)

#对单个jsonl的提取
    for i in c:
        with open("F:\\zsw\\test2\\c\\"+str(cnt)+".c", 'w') as m:
            text = json.loads(i)["func"].replace("\n", "")
            cnt += 1
            m.write(text)
print(cnt)
