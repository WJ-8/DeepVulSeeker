import os
import numpy as np


def slice(save_path, npy):
    train, c = np.split(npy, [int(npy.shape[0] / 10 * 8)], axis=0)
    val, test = np.vsplit(c, [int(c.shape[0] / 2)])
    print(train.shape)
    print(val.shape)
    print(test.shape)
    train_save = save_path + str('train.npy')
    test_save = save_path + str('test.npy')
    val_save = save_path + str('val.npy')
    np.save(train_save, train)
    np.save(test_save, test)
    np.save(val_save, val)


def slice_y(save_path, npy):
    train, c = np.split(npy, [int(npy.shape[0] / 10 * 8)], axis=0)
    val, test = np.hsplit(c, [int(c.shape[0] / 2)])
    print(train.shape)
    print(val.shape)
    print(test.shape)
    train_save = save_path + str('train_y.npy')
    test_save = save_path + str('test_y.npy')
    val_save = save_path + str('val_y.npy')
    np.save(train_save, train)
    np.save(test_save, test)
    np.save(val_save, val)


# 数据集切片
cfg_npy = np.load(r'F:\zsw\cwe_data\cwe754_cfg.npy')
emb_npy = np.load(r'F:\zsw\cwe_data\cwe754_emb.npy')
y_npy = np.load(r'F:\zsw\cwe_data\cwe754_y.npy')
dfg_npy = np.load(r'F:\zsw\cwe_data\cwe754_dfg.npy')
ast_npy = np.load(r'F:\zsw\cwe_data\cwe754_ast.npy')
print(ast_npy.shape)
print(dfg_npy.shape)
print(cfg_npy.shape)
print(emb_npy.shape)
print(y_npy.shape)
slice('F:\\zsw\\cwe_data\\cwenpy\\cwe754\\emb_',emb_npy)
slice('F:\\zsw\\cwe_data\\cwenpy\\cwe754\\ast_',ast_npy)
slice('F:\\zsw\\cwe_data\\cwenpy\\cwe754\\cfg_',cfg_npy)
slice('F:\\zsw\\cwe_data\\cwenpy\\cwe754\\dfg_',dfg_npy)
slice_y('F:\\zsw\\cwe_data\\cwenpy\\cwe754\\',y_npy)


# 已经有分类的数据集
# for i in range(1,10):
#     train_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\ast_train.npy')
#     test_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\ast_test.npy')
#     val_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\ast_val.npy')
#     train1_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\train_cfg.npy')
#     test1_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\test_cfg.npy')
#     val1_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\valid_cfg.npy')
#     train2_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\train_dfg.npy')
#     test2_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\test_dfg.npy')
#     val2_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\valid_dfg.npy')
#     train3_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\train_emb.npy')
#     test3_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\test_emb.npy')
#     val3_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\val_emb.npy')
#     train4_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\train_y.npy')
#     test4_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\test_y.npy')
#     val4_npy = np.load('F:\\zsw\\combine_graph\\cwe78\\'+str(i)+'\\val_y.npy')
#     ast_npy = np.concatenate((test_npy, train_npy), axis=0)
#     ast_npy = np.concatenate((val_npy, ast_npy), axis=0)
#     cfg_npy = np.concatenate((test1_npy, train1_npy), axis=0)
#     cfg_npy = np.concatenate((val1_npy, cfg_npy), axis=0)
#     dfg_npy = np.concatenate((test2_npy, train2_npy), axis=0)
#     dfg_npy = np.concatenate((val2_npy, dfg_npy), axis=0)
#     emb_npy = np.concatenate((test3_npy, train3_npy), axis=0)
#     emb_npy = np.concatenate((val3_npy, emb_npy), axis=0)
#     y_npy = np.concatenate((test4_npy, train4_npy), axis=0)
#     y_npy = np.concatenate((val4_npy, y_npy), axis=0)
#     print(ast_npy.shape)
#     print(cfg_npy.shape)
#     print(dfg_npy.shape)
#     print(emb_npy.shape)
#     print(y_npy.shape)
#     slice('F:\\zsw\\combine_graph\\cwe78\\'+str(i+1)+'\\ast_', ast_npy)
#     slice('F:\\zsw\\combine_graph\\cwe78\\'+str(i+1)+'\\cfg_', cfg_npy)
#     slice('F:\\zsw\\combine_graph\\cwe78\\'+str(i+1)+'\\dfg_', dfg_npy)
#     slice('F:\\zsw\\combine_graph\\cwe78\\'+str(i+1)+'\\emb_', emb_npy)
#     slice_y('F:\\zsw\\combine_graph\\cwe78\\'+str(i+1)+'\\y_', y_npy)
#     path = 'F:\\zsw\\combine_graph\\cwe78\\'+str(i+1)
#     pathlist = os.listdir(path)
#     for j in pathlist:
#         old_name = path + '\\' + j
#         new_name = ''
#         if 'ast' in j:
#             if 'test' in j:
#                 new_name = path + '\\' + 'ast_test.npy'
#             if 'train' in j:
#                 new_name = path + '\\' + 'ast_train.npy'
#             if 'val' in j:
#                 new_name = path + '\\' + 'ast_val.npy'
#         if 'cfg' in j:
#             if 'test' in j:
#                 new_name = path + '\\' + 'test_cfg.npy'
#             if 'train' in j:
#                 new_name = path + '\\' + 'train_cfg.npy'
#             if 'val' in j:
#                 new_name = path + '\\' + 'valid_cfg.npy'
#         if 'dfg' in j:
#             if 'test' in j:
#                 new_name = path + '\\' + 'test_dfg.npy'
#             if 'train' in j:
#                 new_name = path + '\\' + 'train_dfg.npy'
#             if 'val' in j:
#                 new_name = path + '\\' + 'valid_dfg.npy'
#         if 'emb' in j:
#             if 'test' in j:
#                 new_name = path + '\\' + 'test_emb.npy'
#             if 'train' in j:
#                 new_name = path + '\\' + 'train_emb.npy'
#             if 'val' in j:
#                 new_name = path + '\\' + 'val_emb.npy'
#         if '_y' in j:
#             if 'test' in j:
#                 new_name = path + '\\' + 'test_y.npy'
#             if 'train' in j:
#                 new_name = path + '\\' + 'train_y.npy'
#             if 'val' in j:
#                 new_name = path + '\\' + 'val_y.npy'
#         os.rename(old_name, new_name)


