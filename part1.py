import math
from getData import load_data
import numpy as np
import copy
# import matplotlib.pyplot as plt

class TreeNode:
    def __init__(self, left, right, feature, isLeaf, label):
        self.left = left
        self.right = right
        self.feature = feature
        self.isLeaf = isLeaf
        self.label = label


def getCount(data):
    p = 0
    n = 0
    for i in range(0, len(data)):
        if data[i][len(data[0]) - 1] == 1:
            p += 1
        else:
            n += 1
    return p, n


# 先序遍历递归建树
def buildTree(depth, maxdepth, data, f_dict, anti_f_dict, last_f):
    root_f = getRootIndex(anti_f_dict, data)
    pos, neg = getCount(data)
    node = None

    # 说明上一层的节点分配到位 直接作为叶子节点返回
    if pos == 0:
        node = TreeNode(None, None, last_f, True, -1)
        return node

    if neg == 0:
        node = TreeNode(None, None, last_f, True, 1)
        return node

    # 到达最大深度 创建叶子节点返回 不过没影响
    if depth == maxdepth:
        if pos > neg:
            node = TreeNode(None, None, last_f, True, 1)
            return node
        else:
            node = TreeNode(None, None, last_f, True, -1)
            return node


    # 左， 右子节点递归调用所用的数据
    l_data, r_data = getData(root_f, data)

    node = TreeNode(None, None, root_f, False, 0)

    next_f_dict = copy.deepcopy(f_dict)
    next_anti_f_dict = copy.deepcopy(anti_f_dict)

    # 现在用的feature 下一层不能用
    cur_f = anti_f_dict.get(root_f)
    # 现在的feature包含的所有index 都得删了
    f_indices = f_dict.get(cur_f)
    # 一个一个的删掉 O(1)
    for index in f_indices:
        next_anti_f_dict.pop(index)

    # 把feature也删了
    next_f_dict.pop(cur_f)

    # print(depth)
    # print(f_dict)
    node.left = buildTree(depth + 1, maxdepth, l_data, next_f_dict, next_anti_f_dict, root_f)
    # print(depth)
    # print(f_dict)
    node.right = buildTree(depth + 1, maxdepth, r_data, next_f_dict, next_anti_f_dict, root_f)

    return node


def getData(index, data):
    left = []
    right = []
    for i in range(0, len(data)):
        if data[i][index] == 1:
            left.append(data[i])
        else:
            right.append(data[i])
    return np.asarray(left), np.asarray(right)


def getRootIndex(dict_f, data):
    min_gini = 100000000
    root_index = 0
    for feature in dict_f.keys():
        gini = getGini(data, feature)
        if gini < min_gini:
            min_gini = gini
            root_index = feature
    return root_index


# https://www.youtube.com/watch?v=7VeUPuFGJHk&t=695s 13分38秒的方法
def getGini(data, index):
    l_amount = 0
    r_amount = 0
    l_p = 0
    l_n = 0
    r_p = 0
    r_n = 0
    for i in range(0, len(data)):
        if data[i][index] == 1:
            l_amount += 1
            if data[i][len(data[0]) - 1] == 1:
                l_p += 1
            else:
                l_n += 1
        else:
            r_amount += 1
            if data[i][len(data[0]) - 1] == 1:
                r_p += 1
            else:
                r_n += 1

    if r_amount == 0 or l_amount == 0:
        return 1000000
    g_l = 1 - math.pow(l_p / l_amount, 2) - math.pow(l_n / l_amount, 2)
    g_r = 1 - math.pow(r_p / r_amount, 2) - math.pow(r_n / r_amount, 2)
    g = g_l * (l_amount / (l_amount + r_amount)) + g_r * (r_amount / (l_amount + r_amount))
    return g


# 遍历整棵树 直到到叶子节点 直接返回叶子节点的label
def dfs(t_root, instance):
    # 如果访问到叶子节点 就让叶子节点决定命运
    if t_root.isLeaf:
        return t_root.label
    f = t_root.feature
    if instance[f] == 1:
        return dfs(t_root.left, instance)
    else:
        return dfs(t_root.right, instance)


def getAccur(data, t_root):
    correct = 0
    for i in range(0, len(data)):
        res = dfs(t_root, data[i])
        if res == -1:
            if data[i][len(data[0]) - 1] == 0:
                correct += 1
        elif res == 1:
            if data[i][len(data[0]) - 1] == 1:
                correct += 1
    return correct / len(data)


if __name__ == '__main__':
    t_data, t_y, v_data, v_y, f_dict, anti_f_dict = load_data()
    t_y = t_y.reshape(len(t_data), 1)
    v_y = v_y.reshape(len(v_data), 1)
    t_data = np.append(t_data, t_y, axis=1)
    v_data = np.append(v_data, v_y, axis=1)

    maxdepths = [1, 2, 3, 4, 5, 6, 7, 8]


    # plt.figure()
    # plt.xlabel('Height of Trees')
    # plt.ylabel('Validation Accuracy')
    t_as = []
    v_as = []
    for maxdepth in maxdepths:
        root = buildTree(0, maxdepth, t_data, f_dict, anti_f_dict, -1)
        t_accur = getAccur(t_data, root)
        v_accur = getAccur(v_data, root)
        t_as.append(t_accur)
        v_as.append(v_accur)
        print("depth is {:d}".format(maxdepth))
        print("t_accur: {:.13f}".format(t_accur))
        print("v_accur: {:.13f}".format(v_accur))


    # plt.plot(maxdepths, t_as)
    # plt.plot(maxdepths, v_as)
    # plt.show()


