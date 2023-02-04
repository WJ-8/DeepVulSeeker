import json

cnt = 1
with open("../data/raw/FFmpeg/test.jsonl",'r')as f:
    c = f.readlines()
for i in c:
    with open("../data/c/"+str(cnt)+".c", 'w') as m:
        text = json.loads(i)["func"].replace("\n", "")
        cnt += 1
        m.write(text)
print(cnt)
