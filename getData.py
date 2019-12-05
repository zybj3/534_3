import csv
import numpy as np


def load_data():
    t_data = []
    with open('data/pa3_train.csv') as csv_file:
        readCSV = list(csv.reader(csv_file, delimiter=','))
        for row in readCSV:
            t_data.append(row)
    feature = t_data[0]
    f_dict = {}
    anti_f_dict = {}
    for i in range(0, len(feature) - 1):
        true_feature = feature[i][:-2]
        if true_feature in f_dict.keys():
            f_dict[true_feature].append(i)
        else:
            f_dict[true_feature] = []
            f_dict[true_feature].append(i)
        anti_f_dict[i] = true_feature

    t_data = np.asarray(t_data)
    t_data = t_data[1:]
    t_y = t_data[:, 117]
    t_data = np.delete(t_data, -1, axis=1)

    v_data = []
    with open('data/pa3_val.csv') as csv_file:
        readCSV = list(csv.reader(csv_file, delimiter=','))
        for row in readCSV:
            v_data.append(row)

    v_data = np.asarray(v_data)
    v_data = v_data[1:]
    v_y = v_data[:, 117]
    v_data = np.delete(v_data, -1, axis=1)

    t_data = t_data.astype(np.float)
    t_y = t_y.astype(np.float)
    v_data = v_data.astype(np.float)
    v_y = v_y.astype(np.float)

    print(v_data.shape)
    return t_data, t_y, v_data, v_y, f_dict, anti_f_dict


def load_test_data():
    test_data = []
    with open('data/pa3_test.csv') as csv_file:
        readCSV = list(csv.reader(csv_file, delimiter=','))
        for row in readCSV:
            test_data.append(row)
    test_data = np.asarray(test_data)
    test_data = test_data[1:]
    test_data = test_data.astype(np.float)
    print(test_data.shape)
    return test_data


if __name__ == '__main__':
    load_data()
