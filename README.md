# DeepVulSeeker
 implementation of DeepVulSeeker
 ## Data prepara
 Please prepare the jsonl files for the training set, validation set and test set.
The dataset processing file is in `process`  directory
All programs are running in pyCharm

 ### build y

 ```python
 run build_y.py
 ```
 ### build ast
 ```python
run slice_raw.py
run build_dot.py
run build_ast.py
 ```

 ### build cfg and dfg
 ```python
run build_cfgdfg.py
 ```
 ### build pls
 #### get special words
 ```python
run Specialword.py
 ```
 ## train
In the root directory
 ```python
run train.py
 ```
 
 ```
arxiv:
https://arxiv.org/abs/2211.13097
```
