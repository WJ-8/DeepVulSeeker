import subprocess
import os
import json

path_joern = "/home/wj/magic/joern/joern-cli/"
path_train = "/home/wj/magic/data/ast138/"
n = 0
with open("../data/raw/cwe138/cwe-138.jsonl", "r") as f_train:
    data = f_train.readlines()
    for i in data:
        print(n)
        code = json.loads(i)["func"]
        if not os.path.exists(f"{path_train}{n}"):
            os.mkdir(f"{path_train}{n}")
        with open(f"{path_train}{n}/{n}.c", "w") as f:
            f.write(code)
        shell_cpg = f"{path_joern}joern-parse {path_train}{n} --out {path_train}{n}/{n}.bin"
        subprocess.call(shell_cpg, shell=True)
        shell_dot = f"{path_joern}joern-export {path_train}{n}/{n}.bin --repr ast --out {path_train}dot/{n}"
        subprocess.call(shell_dot, shell=True)
        n+=1