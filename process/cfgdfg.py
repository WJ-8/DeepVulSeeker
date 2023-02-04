import json

import numpy as np
import torch
from tree_sitter import Language, Parser
import jsonlines
import collections
from transformers import RobertaTokenizer
import random
import string
import jsonlines

Language.build_library(
    # Store the library in the `build` directory
    'build/my-languages.so',
    # Include one or more languages
    [
        'tree-sitter-c',
    ]
)

C_LANGUAGE = Language('build/my-languages.so', 'c')
parser = Parser()
parser.set_language(C_LANGUAGE)

case_statement = ['case_statement']
non_case_statement = ['labeled_statement', 'compound_statement', 'expression_statement', 'if_statement',
                      'switch_statement', 'do_statement', 'while_statement', 'for_statement', 'return_statement',
                      'break_statement', 'continue_statement', 'goto_statement']


class Node(object):
    def __init__(self, start_point, end_point):
        super(Node, self).__init__()
        self.code = None
        self.next = []
        self.start_point = start_point
        self.end_point = end_point
        self.start = None
        self.end = None


def BFS(nodes):
    d = collections.deque()
    for node in nodes:
        d.append(node)
    rs = []
    while len(d) != 0:
        node = d.popleft()
        if len(node.next) == 0:
            rs.append(node)
        else:
            for item in node.next:
                d.append(item)
    rs = list(set(rs))
    return rs


def find_tail(node):
    if isinstance(node, list):
        return find_tail(node[-1])
    return node


def find_head(node):
    if isinstance(node, list):
        return find_head(node[0])
    return node


def connect_list(nodes):
    for i in range(len(nodes) - 1):
        node = find_tail(nodes[i])
        next_node = find_head(nodes[i + 1])
        if len(node.next) == 0:
            node.next.append(next_node)
        else:
            tail_node = BFS(node.next)
            for item in tail_node:
                item.next.append(next_node)
                item.next = list(set(item.next))
    return nodes


def analyze_if(node, code):
    if_node = Node((-1, -1), (-1, -1))
    has_else = False
    for child in node.children:
        if child.type in ['if', 'parenthesized_expression']:
            if if_node.start_point == (-1, -1):
                if_node.start_point = child.start_point
            if_node.end_point = child.end_point
            continue
        if child.type == 'else':
            has_else = True
            continue
        next_node = find_head(get_control_flow(child, code))
        if next_node is None:
            next_node = Node((-1, -1), (-1, -1))
            next_node.code = 'Empty node'
        if_node.next.append(next_node)
        if_node.next = list(set(if_node.next))
    if not has_else:
        empty = Node((-1, -1), (-1, -1))
        empty.code = 'Empty node'
        if_node.next.append(empty)
        if_node.next = list(set(if_node.next))
    if_node.code = index_to_code_token((if_node.start_point, if_node.end_point), code)
    return if_node


def analyze_while(node, code):
    while_node = Node((-1, -1), (-1, -1))
    for child in node.children:
        if child.type in ['while', 'parenthesized_expression']:
            if while_node.start_point == (-1, -1):
                while_node.start_point = child.start_point
            while_node.end_point = child.end_point
            continue
        next_node = find_head(get_control_flow(child, code))
        if next_node is None:
            next_node = Node((-1, -1), (-1, -1))
            next_node.code = 'Empty node'
        while_node.next.append(next_node)
        while_node.next = list(set(while_node.next))
    while_node.code = index_to_code_token((while_node.start_point, while_node.end_point), code)
    return while_node


def analyze_do_while(node, code):
    do_node = Node((-1, -1), (-1, -1))
    while_node = Node((-1, -1), (-1, -1))
    body = None
    for child in node.children:
        if child.type == 'do':
            if do_node.start_point == (-1, -1):
                do_node.start_point = child.start_point
            do_node.end_point = child.end_point
            continue
        if body is None and child.type != 'parenthesized_expression':
            body = find_head(get_control_flow(child, code))
            if body is None:
                body = Node((-1, -1), (-1, -1))
                body.code = 'Empty node'
            continue
        if child.type in ['while', 'parenthesized_expression', ';']:
            if while_node.start_point == (-1, -1):
                while_node.start_point = child.start_point
            while_node.end_point = child.end_point
            continue
    do_node.next.append(body)
    body.next.append(while_node)
    do_node.code = index_to_code_token((do_node.start_point, do_node.end_point), code)
    while_node.code = index_to_code_token((while_node.start_point, while_node.end_point), code)
    return do_node


def analyze_for(node, code):
    condition = Node((-1, -1), (-1, -1))
    for child in node.children:
        if child.type not in case_statement and child.type not in non_case_statement:
            if condition.start_point == (-1, -1):
                condition.start_point = child.start_point
            condition.end_point = child.end_point
            continue
        next_node = find_head(get_control_flow(child, code))
        if next_node is None:
            next_node = Node((-1, -1), (-1, -1))
            next_node.code = 'Empty node'
        condition.next.append(next_node)
        condition.next = list(set(condition.next))
    condition.code = index_to_code_token((condition.start_point, condition.end_point), code)
    return condition


def analyze_switch(node, code):
    condition = Node((-1, -1), (-1, -1))
    for child in node.children:
        if child.type in ['switch', 'parenthesized_expression']:
            if condition.start_point == (-1, -1):
                condition.start_point = child.start_point
            condition.end_point = child.end_point
            continue
        next_node = get_control_flow(child, code, is_switch_body=True)
        if next_node is None:
            next_node = Node((-1, -1), (-1, -1))
            next_node.code = 'Empty node'
        if not isinstance(next_node, list):
            next_node = [next_node]
        condition.next.extend(next_node)
        condition.next = list(set(condition.next))
    condition.code = index_to_code_token((condition.start_point, condition.end_point), code)
    return condition


def analyze_case(node, code):
    condition = Node((-1, -1), (-1, -1))
    flag = 0
    for child in node.children:
        if flag == 0:
            if condition.start_point == (-1, -1):
                condition.start_point = child.start_point
            condition.end_point = child.end_point
            if child.type == ':':
                flag = 1
            continue
        next_node = find_head(get_control_flow(child, code))
        if next_node is None:
            next_node = Node((-1, -1), (-1, -1))
            next_node.code = 'Empty node'
        condition.next.append(next_node)
        condition.next = list(set(condition.next))
    condition.code = index_to_code_token((condition.start_point, condition.end_point), code)
    return condition


def get_control_flow(root_node, code, is_switch_body=False):
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        return None
    else:
        if root_node.start_point[0] == root_node.end_point[0]:
            node = Node(root_node.start_point, root_node.end_point)
            node.code = index_to_code_token((root_node.start_point, root_node.end_point), code)
            return node
        nodes = []
        for child in root_node.children:
            if child.type == 'if_statement':
                nodes.append(analyze_if(child, code))
            elif child.type == 'while_statement':
                nodes.append(analyze_while(child, code))
            elif child.type == 'do_statement':
                nodes.append(analyze_do_while(child, code))
            elif child.type == 'for_statement':
                nodes.append(analyze_for(child, code))
            elif child.type == 'switch_statement':
                nodes.append(analyze_switch(child, code))
            elif child.type == 'case_statement':
                nodes.append(analyze_case(child, code))
            elif child.type in ['{', '}']:
                continue
            else:
                next_node = find_head(get_control_flow(child, code))
                if next_node is not None:
                    nodes.append(next_node)
        if len(nodes) == 0:
            return None
        if not is_switch_body:
            nodes = connect_list(nodes)
        return nodes


def tree_to_token_index(root_node):
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens


def index_to_code_token(index, code):
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:] + ' \n '
        for i in range(start_point[0] + 1, end_point[0]):
            s += code[i].strip() + ' \n '
        s += code[end_point[0]][:end_point[1]].strip()
    return s


def get_code_string(code):
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    code_tokens = []
    last_row = 0
    for index in tokens_index:
        cur_start_row = index[0][0]
        cur_end_row = index[1][0]
        cur_code = index_to_code_token(index, code)
        if cur_start_row != last_row:
            cur_code = "\n " + cur_code
        last_row = cur_end_row
        code_tokens.append(cur_code)
    return ' '.join(code_tokens)


def get_token_position(node, code_line):
    line_id = int(node.start_point[0])
    start_pos = int(node.start_point[1])
    end_pos = int(node.end_point[1])
    token_num = len(node.code.split(' '))
    before_num = line_id
    for i in range(line_id):
        line = code_line[i].strip()
        before_num += len(line.split(' '))
    line = code_line[line_id][:start_pos].strip()
    if len(line) != 0:
        before_num += len(code_line[line_id][:start_pos].strip().split(' '))
    return before_num, before_num + token_num - 1


def get_path(node, code_list, result):
    if node.start_point != (-1, -1):
        code_list.append(node)
    if len(node.next) == 0:
        result.append(code_list.copy())
        return result
    else:
        for item in node.next:
            result = get_path(item, code_list.copy(), result)
        return result


def create_mask(code, token_pos, paths, max_source_length, last_idx, eos_position, statement_path, df_path):
    code = code.split('\n')
    mask = torch.zeros((max_source_length, max_source_length), dtype=torch.int)
    mask[0, :eos_position + 1] = 1
    mask[:eos_position + 1, 0] = 1
    mask[eos_position, :eos_position + 1] = 1
    mask[:eos_position + 1, eos_position] = 1
    for path in paths:
        path_len = len(path)
        for i in range(path_len - 1):
            start_node = path[i]
            start_node_s, start_node_e = get_token_position(start_node, code)
            # start_node_s += 1
            # start_node_e += 1
            for j in range(i + 1, min(path_len, i + 2)):
                end_node = path[j]
                end_node_s, end_node_e = get_token_position(end_node, code)
                # end_node_s += 1
                # end_node_e += 1
                for head_index in range(start_node_s, start_node_e + 1):
                    if head_index > last_idx:
                        break
                    head_begin, head_end = token_pos[head_index]
                    if head_begin == -1 and head_end == -1:
                        continue
                    for tail_index in range(end_node_s, end_node_e + 1):
                        if tail_index > last_idx:
                            break
                        tail_begin, tail_end = token_pos[tail_index]
                        if tail_begin == -1 and tail_end == -1:
                            continue
                        mask[head_begin: head_end + 1, tail_begin: tail_end + 1] = 1
                        mask[tail_begin: tail_end + 1, head_begin: head_end + 1] = 1

    mask_statement = torch.zeros((max_source_length, max_source_length), dtype=torch.int)
    mask_statement[0, :eos_position + 1] = 2
    mask_statement[:eos_position + 1, 0] = 2
    mask_statement[eos_position, :eos_position + 1] = 2
    mask_statement[:eos_position + 1, eos_position] = 2
    for node in statement_path:
        node_s, node_e = get_token_position(node, code)
        # node_s += 1
        # node_e += 1
        for head_index in range(node_s, node_e + 1):
            if head_index > last_idx:
                break
            head_begin, head_end = token_pos[head_index]
            if head_begin == -1 and head_end == -1:
                continue
            for tail_index in range(node_s, node_e + 1):
                if tail_index > last_idx:
                    break
                tail_begin, tail_end = token_pos[tail_index]
                if tail_begin == -1 and tail_end == -1:
                    continue
                mask_statement[head_begin: head_end + 1, tail_begin: tail_end + 1] = 2
                mask_statement[tail_begin: tail_end + 1, head_begin: head_end + 1] = 2

    mask_df = torch.zeros((max_source_length, max_source_length), dtype=torch.int)
    mask_df[0, :eos_position + 1] = 4
    mask_df[:eos_position + 1, 0] = 4
    mask_df[eos_position, :eos_position + 1] = 4
    mask_df[:eos_position + 1, eos_position] = 4

    for path in df_path:
        if len(path[3]) == 0:
            continue
        head_index = path[1]
        if head_index > last_idx:
            break
        head_begin, head_end = token_pos[head_index]
        if head_begin == -1 and head_end == -1:
            continue
        for tail_index in path[3]:
            if tail_index > last_idx:
                break
            tail_begin, tail_end = token_pos[tail_index]
            if tail_begin == -1 and tail_end == -1:
                continue
            mask_df[head_begin: head_end + 1, tail_begin: tail_end + 1] = 4
            mask_df[tail_begin: tail_end + 1, head_begin: head_end + 1] = 4

    mask = mask + mask_statement + mask_df
    return mask


""" def get_code_mask(code):
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node
    nodes = get_control_flow(root_node, code.split('\n'))
    head = find_head(nodes)
    code_list = []
    result = []
    result = get_path(head, code_list, result)
    statement_result = []
    get_statement(root_node, statement_result, code.split('\n'))
    for path in result:
        for node in path:
            s, e = get_token_position(node, code.split('\n'))
            node.start = s
            node.end = e
    df_path, states = get_data_flow(root_node, {}, code.split('\n'))
    mask = create_mask(code,result, max_source_length, last_idx,statement_result, df_path)
    return mask """


def get_statement(root_node, result, code):
    if root_node.start_point[0] == root_node.end_point[0]:
        node = Node(root_node.start_point, root_node.end_point)
        node.code = index_to_code_token((root_node.start_point, root_node.end_point), code)
        result.append(node)
    else:
        for child in root_node.children:
            get_statement(child, result, code)


def if_data_flow(root_node, states, code_line):
    tag = False
    result = []
    other_states = []
    for child in root_node.children:
        if child.type == 'else':
            tag = True

    for child in root_node.children:
        if child.type in ['if', 'else']:
            continue
        if child.type == 'parenthesized_expression' or tag is False:
            tmp, states = get_data_flow(child, states, code_line)
            result.extend(tmp)
        else:
            tmp, tmp_states = get_data_flow(child, states.copy(), code_line)
            result.extend(tmp)
            other_states.append(tmp_states)

    if tag is False:
        other_states.append(states)
    new_states = {}
    for dic in other_states:
        for key in dic:
            if key not in new_states:
                new_states[key] = dic[key].copy()
            else:
                new_states[key] += dic[key]
    for key in new_states:
        new_states[key] = list(set(new_states[key]))
    return result, new_states


def while_data_flow(root_node, states, code_line):
    result = []
    for child in root_node.children:
        if child.type == 'while':
            continue
        tmp, states = get_data_flow(child, states, code_line)
        result.extend(tmp)
    return result, states


def do_data_flow(root_node, states, code_line):
    result = []
    for child in root_node.children:
        if child.type in ['while', 'do']:
            continue
        tmp, states = get_data_flow(child, states, code_line)
        result.extend(tmp)
    return result, states


def for_data_flow(root_node, states, code_line):
    result = []
    for child in root_node.children:
        if child.type in ['for', '(', ')']:
            continue
        tmp, states = get_data_flow(child, states, code_line)
        result.extend(tmp)
    return result, states


def find_variable(root_node, code_line):
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        index = (root_node.start_point, root_node.end_point)
        code = index_to_code_token(index, code_line)
        if root_node.type != code:
            return [(root_node.start_point, root_node.end_point)]
        else:
            return []
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += find_variable(child, code_line)
        return code_tokens


def declaration_data_flow(root_node, states, code_line):
    result = []

    nodes = [x for x in root_node.child_by_field_name('declarator').children if x.type != ',']
    if len(nodes) == 0:
        for child in root_node.children:
            nodes.append(child)
    left_nodes = []
    right_nodes = []
    flag = False
    for node in nodes:
        if node.type == '=':
            flag = True
            continue
        if flag:
            right_nodes.append(node)
        else:
            left_nodes.append(node)

    tag = False

    for left_node, right_node in zip(left_nodes, right_nodes):
        tag = True
        left_tokens_index = find_variable(left_node, code_line)
        right_tokens_index = find_variable(right_node, code_line)
        tmp = []
        for token1_index in left_tokens_index:
            code1 = index_to_code_token(token1_index, code_line)
            node1 = Node(token1_index[0], token1_index[1])
            node1.code = code1
            idx1 = get_token_position(node1, code_line)[0]
            for x in right_tokens_index:
                code2 = index_to_code_token(x, code_line)
                node2 = Node(x[0], x[1])
                node2.code = code2
                idx2 = get_token_position(node2, code_line)[0]
                tmp.append((code1, idx1, 'computedFrom', [idx2]))
                if code2 not in states:
                    tmp.append((code2, idx2, 'from', []))
                else:
                    tmp.append((code2, idx2, 'from', states[code2]))
            states[code1] = [idx1]
            result.extend(tmp)
    if tag is False:
        for left_node in left_nodes:
            left_tokens_index = find_variable(left_node, code_line)
            tmp = []
            for token1_index in left_tokens_index:
                code1 = index_to_code_token(token1_index, code_line)
                node1 = Node(token1_index[0], token1_index[1])
                node1.code = code1
                idx1 = get_token_position(node1, code_line)[0]
                tmp.append((code1, idx1, 'from', []))
                states[code1] = [idx1]
            result.extend(tmp)

    return result, states


def assignment_data_flow(root_node, states, code_line):
    result = []
    left_nodes = [x for x in root_node.child_by_field_name('left').children if x.type != ',']
    right_nodes = [x for x in root_node.child_by_field_name('right').children if x.type != ',']
    if len(right_nodes) != len(left_nodes):
        left_nodes = [root_node.child_by_field_name('left')]
        right_nodes = [root_node.child_by_field_name('right')]
    if len(left_nodes) == 0:
        left_nodes = [root_node.child_by_field_name('left')]
    if len(right_nodes) == 0:
        right_nodes = [root_node.child_by_field_name('right')]

    for node in right_nodes:
        tmp, states = get_data_flow(node, states, code_line)
        result.extend(tmp)

    for left_node, right_node in zip(left_nodes, right_nodes):
        left_tokens_index = find_variable(left_node, code_line)
        right_tokens_index = find_variable(right_node, code_line)
        tmp = []
        for token1_index in left_tokens_index:
            code1 = index_to_code_token(token1_index, code_line)
            node1 = Node(token1_index[0], token1_index[1])
            node1.code = code1
            idx1 = get_token_position(node1, code_line)[0]
            for x in right_tokens_index:
                code2 = index_to_code_token(x, code_line)
                node2 = Node(x[0], x[1])
                node2.code = code2
                idx2 = get_token_position(node2, code_line)[0]
                tmp.append((code1, idx1, 'computedFrom', [idx2]))
            states[code1] = [idx1]
            result.extend(tmp)
    return result, states


def get_data_flow(root_node, states, code_line):
    # code: [, , , , ]
    states = states.copy()
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        node = Node(root_node.start_point, root_node.end_point)
        code = index_to_code_token((root_node.start_point, root_node.end_point), code_line)
        node.code = code
        begin, end = get_token_position(node, code_line)
        if root_node.type == code:
            return [], states
        elif code in states:
            return [(code, begin, 'from', states[code].copy())], states
        else:
            if root_node.type == 'identifier':
                states[code] = [begin]
            return [(code, begin, 'from', [])], states
    else:
        result = []
        for child in root_node.children:
            if child.type == 'if_statement':
                tmp, states = if_data_flow(child, states, code_line)
                result.extend(tmp)
            elif child.type == 'while_statement':
                tmp, states = while_data_flow(child, states, code_line)
                result.extend(tmp)
            elif child.type == 'assignment_expression':
                tmp, states = assignment_data_flow(child, states, code_line)
                result.extend(tmp)
            elif child.type == 'declaration':
                tmp, states = declaration_data_flow(child, states, code_line)
                result.extend(tmp)
            else:
                tmp, states = get_data_flow(child, states, code_line)
                result.extend(tmp)

        return result, states


# 格式化一下代码
def get_str_code(code):
    code_list = list(code)
    for i in range(0, len(code_list) - 1):
        if code_list[i] == ";" and code_list[i + 1] != "\n":
            code_list[i] = ";" + "\n"
        if code_list[i] == "{" and code_list[i + 1] != "\n":
            code_list[i] = "{" + "\n"
        if code_list[i + 1] == "}" and code_list[i] != "\n":
            code_list[i] += "\n"
        if code_list[i] == ' ' and code_list[i + 1] == ";":
            code_list[i] = ''
    return ''.join(code_list)


# ——————————————————————构建dfs邻接矩阵————————————————————————————
def create_matrix(df_path):
    n = int(len(df_path))
    matrix = np.ones((200, 200)) * (-1)
    temp1 = df_path[0][1]
    index1 = 0
    if n > 200:
        n = 200
    for i in range(0, n):
        temp2 = df_path[i][1]
        index2 = i
        # matrix[i][i] = 1
        if temp2 != temp1:
            temp1 = df_path[i][1]
            index1 = i
        if temp2 == temp1:
            matrix[index1][index2] = 1
            matrix[index2][index1] = 1
        nn = df_path[i][3]
        if nn:
            if nn[0] >= temp2:
                for t1 in range(index2, n):
                    temp3 = df_path[t1][1]
                    index3 = t1
                    if temp3 == nn[0]:
                        matrix[index3][index2] = 1
                        matrix[index2][index3] = 1
            elif nn[0] < temp2:
                for t2 in range(0, index2):
                    temp3 = df_path[t2][1]
                    index3 = t2
                    if temp3 == nn[0]:
                        matrix[index3][index2] = 1
                        #增添
                        matrix[index2][index3] = 1
    return matrix


def create_dfs_print_matrix(filename):
    tree = parser.parse(bytes(filename, "utf8"))
    test_root_node = tree.root_node
    test_nodes = get_control_flow(test_root_node, filename.split('\n'))
    test_head = find_head(test_nodes)
    code_list = []
    result = []
    result = get_path(test_head, code_list, result)
    statement_result = []
    get_statement(test_root_node, statement_result, filename.split('\n'))
    for path in result:
        for node in path:
            s, e = get_token_position(node, filename.split('\n'))
            node.start = s
            node.end = e
    df_path, states = get_data_flow(test_root_node, {}, filename.split('\n'))
    """———————————显示所有节点信息——————————————
    for i in range(0, int(len(df_path))):
        print(df_path[i])
    print(test_root_node)
    print(test_nodes)"""
    return df_path


def find_node_cfg(test_root_node, total_number):
    global mask_cfg
    global mask_cfg_1
    child_number = int(test_root_node.child_count)
    test_node = test_root_node
    if child_number != 0:
        for i in range(0, child_number):
            total_number, mask_cfg, mask_cfg_1 = find_node_cfg(test_node.children[i], total_number)
    if child_number == 0:
        xx = {total_number: test_node.start_point}
        yy = {test_node.start_point: total_number}
        mask_cfg.update(xx)
        mask_cfg_1.update(yy)
        total_number += 1
    return total_number, mask_cfg, mask_cfg_1


mask_cfg = {}
mask_cfg_1 = {}


def bl(start, end, matrix):
    for i in range(start, end):
        for j in range(start, end):
            matrix[i][j] = 1


def all_next_node_cfg(test_node):
    node_number = test_node.child_count
    temp1 = test_mask_cfg_1[test_node.start_point]
    if temp1 >= 200:
        return None
    if node_number != 0 and temp1 < 200:
        for i in range(0, node_number):
            all_next_node_cfg(test_node.children[i])
    return test_mask_cfg_1[test_node.start_point]


def connect_node_cfg(matrix, test_node):
    node_number = test_node.child_count
    temp2 = test_mask_cfg_1[test_node.start_point]
    if node_number != 0 and temp2 < 200:
        for i in range(0, node_number):
            temp3 = test_mask_cfg_1[test_node.children[i].start_point]
            if temp3 < 200:
                connect_node_cfg(matrix, test_node.children[i])
    if test_node.next_sibling is not None:
        temp1 = test_mask_cfg_1[test_node.next_sibling.start_point]
        if temp1 < 200 and temp2 < 200:
            temp4 = all_next_node_cfg(test_node.next_sibling)
            if temp4 is not None:
                matrix[test_mask_cfg_1[test_node.start_point]][temp4] = 1
                #增添
                matrix[temp4][test_mask_cfg_1[test_node.start_point]] = 1


def create_cfg_matrix():
    cfg_matrix = np.ones((200, 200)) * (-1)
    connect_node_cfg(cfg_matrix, test_root_node)
    return cfg_matrix


# 输入 filename为code：
# filename = """int a=5;int c=a+b;if(c>4)c=1;return 0 ;"""
# filename = get_str_code(filename)
# create_dfs_print_matrix(filename)
# jsonl文件

with open("F:\\zsw\\test2\\test.jsonl", "r+", encoding="utf8") as f:
    c = f.readlines()
    time = 0
    for i in c:
        time += 1
        text = json.loads(i)["func"].replace("\n", "")
        filename = text
        # filename = get_str_code(filename)
        # create_dfs_print_matrix(filename)
        tree = parser.parse(bytes(filename, "utf8"))
        test_root_node = tree.root_node
        # print(filename)
        test_nodes = get_control_flow(test_root_node, filename.split('\n'))
        test_node_number, test_mask_cfg, test_mask_cfg_1 = find_node_cfg(test_root_node, 0)
        cfg_matrix = create_cfg_matrix()
        df_path = create_dfs_print_matrix(filename)
        dfg_matrix = create_matrix(df_path)
        dfg_matrix = np.expand_dims(dfg_matrix, axis=0)
        cfg_matrix = np.expand_dims(cfg_matrix, axis=0)
        if time == 1:
            ddfg = dfg_matrix
            ccfg = cfg_matrix
        elif time > 1:
            ddfg = np.concatenate((ddfg,dfg_matrix),axis=0)
            ccfg = np.concatenate((ccfg,cfg_matrix),axis=0)
        print(ccfg.shape)
        print(ddfg.shape)
    np.save('F:\\zsw\\test2\\npy\\test_cfg.npy',ccfg)
    np.save('F:\\zsw\\test2\\npy\\test_dfg.npy',ddfg)
f.close()

# tree = parser.parse(bytes(filename, "utf8"))
# test_root_node = tree.root_node
# # print(filename)
# test_nodes = get_control_flow(test_root_node, filename.split('\n'))
#
# test_node_number, test_mask_cfg, test_mask_cfg_1 = find_node_cfg(test_root_node, 0)
# # print(test_node_number)
#
# tree = parser.parse(bytes(filename, "utf8"))
# test_root_node = tree.root_node
# # print(filename)
# test_nodes = get_control_flow(test_root_node, filename.split('\n'))

# print(test_mask_cfg)
# print(test_mask_cfg_1)
# print(create_cfg_matrix(test_node_number))
