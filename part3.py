import csv
import math
from getData import load_data
from getData import load_test_data
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
        if data[i][len(data[0]) - 2] == 1:
            p += 1
        else:
            n += 1
    return p, n


def buildTree(depth, maxdepth, data, f_dict_c, anti_f_dict_c, last_f, f_set):
    if len(data) == 0:
        return TreeNode(None, None, -1, True, -1)
    root_f = getRootIndex(anti_f_dict_c, data)
    pos, neg = getCount(data)
    node = None
    if pos == 0:
        node = TreeNode(None, None, last_f, True, -1)
        return node

    if neg == 0:
        node = TreeNode(None, None, last_f, True, 1)
        return node

    if depth == maxdepth:
        if pos > neg:
            node = TreeNode(None, None, last_f, True, 1)
            return node
        else:
            node = TreeNode(None, None, last_f, True, -1)
            return node

    l_data, r_data = getData(root_f, data)
    node = TreeNode(None, None, root_f, False, 0)
    f_set.add(root_f)

    next_f_dict = copy.deepcopy(f_dict_c.copy())
    next_anti_f_dict = copy.deepcopy(anti_f_dict_c.copy())

    cur_f = anti_f_dict_c.get(root_f)
    f_indices = next_f_dict.get(cur_f)
    for index in f_indices:
        next_anti_f_dict.pop(index)

    next_f_dict.pop(cur_f)
    node.left = buildTree(depth + 1, maxdepth, l_data, next_f_dict, next_anti_f_dict, root_f, f_set)
    node.right = buildTree(depth + 1, maxdepth, r_data, next_f_dict, next_anti_f_dict, root_f, f_set)
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
    min_error = 100000000
    root_index = 0
    for feature in dict_f.keys():
        error = getError(data, feature)
        if error < min_error:
            min_error = error
            root_index = feature
    return root_index


def getError(data, f_index):
    error = 0
    for i in range(0, len(data)):
        if data[i][f_index] == 1:
            if data[i][len(data[0]) - 2] == 0:
                error += data[i][len(data[0]) - 1]
        else:
            if data[i][len(data[0]) - 2] == 1:
                error += data[i][len(data[0]) - 1]
    return error


def dfs(t_root, instance):
    if t_root.isLeaf:
        return t_root.label
    f = t_root.feature
    if instance[f] == 1:
        return dfs(t_root.left, instance)
    else:
        return dfs(t_root.right, instance)


def updateWeight(t_data_c, root, alphas):
    amount_of_error, predict_vals = getAmountOfError(t_data_c, root)
    alpha = 0.5 * math.log((1 - amount_of_error) / amount_of_error)
    alphas[root] = alpha
    for i in range(0, len(t_data_c)):
        if t_data_c[i][len(t_data_c[0]) - 2] == predict_vals[i]:
            t_data_c[i][len(t_data_c[0]) - 1] *= math.exp(-1 * alpha)
        else:
            t_data_c[i][len(t_data_c[0]) - 1] *= math.exp(alpha)
    sum = 0
    for i in range(0, len(t_data_c)):
        sum += t_data_c[i][len(t_data_c[0]) - 1]

    for i in range(0, len(t_data_c)):
        t_data_c[i][len(t_data_c[0]) - 1] /= sum


def getAmountOfError(t_data_c, root):
    a_error = 0
    predict_vals = []
    for i in range(0, len(t_data_c)):
        predict = dfs(root, t_data_c[i])
        if predict == -1:
            predict_vals.append(0)
            if t_data_c[i][len(t_data_c[0]) - 2] == 1:
                a_error += t_data_c[i][len(t_data_c[0]) - 1]
        else:
            predict_vals.append(1)
            if t_data_c[i][len(t_data_c[0]) - 2] == 0:
                a_error += t_data_c[i][len(t_data_c[0]) - 1]
    return a_error, predict_vals


def train(t_data_c, v_data_c, f_dict_c, anti_f_dict_c, l, h):
    alphas = {}
    f_used = set()

    f_d = copy.deepcopy(f_dict_c)
    anti_f_d = copy.deepcopy(anti_f_dict_c)

    t_data_c_c = copy.deepcopy(t_data_c)

    for i in range(0, l):
        root = buildTree(0, h, t_data_c_c, f_d, anti_f_d, -1, f_used)
        updateWeight(t_data_c_c, root, alphas)
        for used_f in f_used:
            if used_f in anti_f_d.keys():
                f = anti_f_d[used_f]
                f_list = f_d.get(f)
                for brother in f_list:
                    if brother in anti_f_d:
                        anti_f_d.pop(brother)
                if f in f_d:
                    f_d.pop(f)
    print("L is {:d}".format(l) + " Height is {:d}".format(h))
    t_acc = getAccur(alphas, t_data_c_c, 2)
    print('t_acc is :{:.3f}'.format(t_acc))
    v_acc = getAccur(alphas, v_data_c, 1)
    print('v_acc is: {:.3f}'.format(v_acc))
    return t_acc, v_acc, alphas


def getAccur(alphas, t_data_c_c, offset):
    predict_v = []
    for i in range(0, len(t_data_c_c)):
        predict_v.append(0)

    for root in alphas.keys():
        for i in range(0, len(t_data_c_c)):
            predict = dfs(root, t_data_c_c[i])
            predict_v[i] += predict * alphas.get(root)

    correct = 0
    for i in range(0, len(t_data_c_c)):
        if predict_v[i] > 0:
            if t_data_c_c[i][len(t_data_c_c[0]) - offset] == 1:
                correct += 1
        else:
            if t_data_c_c[i][len(t_data_c_c[0]) - offset] == 0:
                correct += 1

    return correct / len(t_data_c_c)


def getPredict(test_data, alphas):
    p = []
    for i in range(0, len(test_data)):
        p.append(0)

    for root in alphas.keys():
        for i in range(0, len(test_data)):
            predict = dfs(root, test_data[i])
            p[i] += predict * alphas.get(root)
    for i in range(0, len(p)):
        if p[i] < 0:
            p[i] = 0
        else:
            p[i] = 1
    return p


if __name__ == '__main__':
    L = [1, 2, 5, 10, 15]
    t_data, t_y, v_data, v_y, f_dict, anti_f_dict = load_data()
    weight = np.ones(len(t_data))
    weight /= len(t_data)
    weight = weight.reshape(len(t_data), 1)
    t_y = t_y.reshape(len(t_data), 1)
    t_data = np.append(t_data, t_y, axis=1)
    t_data = np.append(t_data, weight, axis=1)
    v_y = v_y.reshape(len(v_y), 1)
    v_data = np.append(v_data, v_y, axis=1)

    for l in L:
        t_acc, v_acc, alphas = train(t_data, v_data, f_dict, anti_f_dict, l, 1)

    # test_data = load_test_data()
    # p = getPredict(test_data, alphas)
    #
    # with open('predict.csv', mode='w') as res:
    #     res_writer = csv.writer(res, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #     for predit in p:
    #         res_writer.writerow([str(predit)])
    #     res.close()
    # print(p)
