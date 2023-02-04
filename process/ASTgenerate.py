from sklearn import tree
import string
import numpy as np
import os

path = r"F:\zsw\test2\c"
fileList = os.listdir(path)  # 待修改文件夹
os.chdir(path)  # 将当前工作目录修改为待修改文件夹的位置
num = 0  # 名称变量
for fileName in fileList:  # 遍历文件夹中所有文件
    num += 1
print(num)

# nummm = 0
# white_list = []
# for j in range(0, num):
#     path_ast = 'C:\\Users\\wangj\\Desktop\\data_sard\\ast_dot\\cwe404\\' + str(j)
#     if not os.listdir(path_ast):
#         nummm += 1
#         white_list.append(j)
#     f = open("C:\\Users\\wangj\\Desktop\\graph_cwe\\cwe404\\white_file_val.txt", "w+")
#     f.write(",".join(str(i) for i in white_list))
# print(nummm)

# time = 0
# for j in range(0, num):
#     ast_matrix = np.ones((200, 200)) * (-1)
#     path_ast = 'C:\\Users\\wangj\\Desktop\\data_sard\\ast_dot\\cwe404\\' + str(j)
#     time += 1
#     if os.listdir(path_ast):
#         path_ast = 'C:\\Users\\wangj\\Desktop\\data_sard\\ast_dot\\cwe404\\' + str(j) + '\\0-ast.dot'
#         print(path_ast)
#         with open(path_ast, "r", encoding='utf8') as f:
#             c = f.readlines()
#             fin = 0
#             head_node = c[1][0:9]
#             print(head_node)
#             for i in c:
#                 if i[0:11] == '  '+head_node:  # 找节点
#                     fin += 1
#                 if fin > 1:
#                     if i != c[-1]:
#                         m = int(i[7]) * 100 + int(i[8]) * 10 + int(i[9]) - 100
#                         m1 = int(i[20]) * 100 + int(i[21]) * 10 + int(i[22]) - 100
#                         if m < 200 and m1 < 200:
#                             ast_matrix[m][m1] = 1
#                             ast_matrix[m1][m] = 1
#                 elif fin <= 1 and i != c[0]:
#                     s = int(i[5]) * 100 + int(i[6]) * 10 + int(i[7]) - 100
#                     if s < 200:
#                         ast_matrix[s][s] = 1
#         f.close()
#     ast_matrix = np.expand_dims(ast_matrix, 0)
#     if time == 1:
#         ast1 = ast_matrix
#     if time > 1:
#         ast1 = np.concatenate((ast1, ast_matrix), axis=0)
#     print(ast1.shape)
# np.save('C:\\Users\\wangj\\Desktop\\graph_cwe\\cwe404\\ast_cwe404.npy', ast1)
time = 0

n = 0
ast_matrix = np.ones((num, 200, 200)) * (-1)
for j in range(0, num):
    path_ast = 'F:\\zsw\\test2\\astffmpeg\\dot\\' + str(j)
    time += 1
    if os.listdir(path_ast):
        path_ast = 'F:\\zsw\\test2\\astffmpeg\\dot\\' + str(j) + '\\0-ast.dot'
        print(path_ast)
        with open(path_ast, "r", encoding='utf8') as f:
            c = f.readlines()
            fin = 0
            head_node = c[1][0:9]
            print(head_node)
            n = 0
            for i in c:
                if i[0:11] == '  ' + head_node:  # 找节点
                    fin += 1
                if fin > 1:
                    if i != c[-1]:
                        m = int(i[7]) * 100 + int(i[8]) * 10 + int(i[9]) - 100
                        m1 = int(i[20]) * 100 + int(i[21]) * 10 + int(i[22]) - 100
                        if m < 200 and m1 < 200:
                            ast_matrix[j][m][m1] = 1
                            ast_matrix[j][m1][m] = 1
                            # print(1,end=',')
                            # n += 1
                elif fin <= 1 and i != c[0]:
                    s = int(i[5]) * 100 + int(i[6]) * 10 + int(i[7]) - 100
                    if s < 200:
                        ast_matrix[j][s][s] = 1
                        n += 1
                # print(n)
        f.close()
print(ast_matrix.shape)
np.save('F:\\zsw\\test2\\npy\\ast_test.npy', ast_matrix)

# ast_matrix = np.ones((200, 200)) * (-1)
# path_ast = 'C:\\Users\\wangj\\Desktop\\dot_qemu\\val\\8166\\0-ast.dot'
# with open(path_ast, "r", encoding='utf8') as f:
#         c = f.readlines()
#         fin = 0
#         for i in c:
#             print(i)
#             if i[0:11] == '  "1000100"':  # 找节点
#                 fin += 1
#             if fin > 1:
#                 if i != c[-1]:
#                     m = int(i[7]) * 100 + int(i[8]) * 10 + int(i[9]) - 100
#                     m1 = int(i[20]) * 100 + int(i[21]) * 10 + int(i[22]) - 100
#                     if m < 200 and m1 < 200:
#                         ast_matrix[m][m1] = 1
#                         ast_matrix[m1][m] = 1
#                     elif fin <= 1 and i != c[0]:
#                         s = int(i[5]) * 100 + int(i[6]) * 10 + int(i[7]) - 100
#                         if s < 200:
#                             ast_matrix[s][s] = 1
#             print(ast_matrix)
