import os

# path = r'F:\zsw\fundd_data'
path =r'F:\zsw\testdata_new'
pathlist = os.listdir(path)
for i in pathlist:
    old_name = path + '\\' + i
    new_name = ''
    if 'ast' in i:
        if 'test' in i:
            new_name = path + '\\' + 'ast_test.npy'
        if 'train' in i:
            new_name = path + '\\' + 'ast_train.npy'
        if 'val' in i:
            new_name = path + '\\' + 'ast_val.npy'
    if 'cfg' in i:
        if 'test' in i:
            new_name = path + '\\' + 'test_cfg.npy'
        if 'train' in i:
            new_name = path + '\\' + 'train_cfg.npy'
        if 'val' in i:
            new_name = path + '\\' + 'valid_cfg.npy'
    if 'dfg' in i:
        if 'test' in i:
            new_name = path + '\\' + 'test_dfg.npy'
        if 'train' in i:
            new_name = path + '\\' + 'train_dfg.npy'
        if 'val' in i:
            new_name = path + '\\' + 'valid_dfg.npy'
    if 'emb' in i:
        if 'test' in i:
            new_name = path + '\\' + 'test_emb.npy'
        if 'train' in i:
            new_name = path + '\\' + 'train_emb.npy'
        if 'val' in i:
            new_name = path + '\\' + 'val_emb.npy'
    if '_y' in i:
        if 'test' in i:
            new_name = path + '\\' + 'test_y.npy'
        if 'train' in i:
            new_name = path + '\\' + 'train_y.npy'
        if 'val' in i:
            new_name = path + '\\' + 'val_y.npy'
    os.rename(old_name, new_name)
