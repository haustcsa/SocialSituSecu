import os
import csv
import numpy as np

Invalid_Sign = -1


def encoding(s, dict, can_add_new=True):
    if not can_add_new:
        if s not in dict:
            return None
    if s in dict:
        return dict[s]
    new_num = len(dict)
    dict[s] = new_num
    return new_num


def load_data(data_file_name, truth_file_name, IsContinuous):
    M = 0
    L = 0
    source_dict = {}
    obj_dict = {}
    value_dict = {}
    with open(data_file_name, encoding='utf-8', errors='ignore', mode='r') as fp1:
        reader = csv.reader(fp1)
        for row in reader:
            M = len(row) - 2
            if len(row[0]) > 0:
                # 统计有几个数据源
                encoding(int(row[0]), source_dict)
            if len(row[-1]) > 0:
                # 统计有几个事件,并编号{7: 0, 9: 1, 12: 2, 23: 3, 25: 4, 28: 5, 30: 6, 31: 7}
                encoding(int(row[-1]), obj_dict)
    L = len(obj_dict)
    K = len(source_dict)
    # print(source_dict, obj_dict)
    # K数据源数量   L事实数量   M属性数量
    print('K=', K, '  L=', L, '  M=', M)
    data_mat = np.ones(shape=(K, L, M), dtype=np.double) * Invalid_Sign
    truth_mat = np.ones(shape=(L, M), dtype=np.double) * Invalid_Sign
    valid_mat = np.zeros(shape=(L, M), dtype=bool)
    with open(data_file_name, errors='ignore', mode='r') as fp1:
        reader = csv.reader(fp1)
        for row in reader:
            if len(row[0]) == 0:
                continue
            if len(row[-1]) == 0:
                continue
            # print(source_dict, obj_dict)
            # k,l返回的是数据源和事实的位置
            k, l = encoding(int(row[0]), source_dict), encoding(int(row[-1]), obj_dict)
            # print('k, l:', k, l)
            for m in range(1, len(row) - 1):
                if len(row[m]) == 0:
                    continue
                if IsContinuous[m - 1]:  # 判断是否是连续型数据
                    data_mat[k, l, m - 1] = np.double(row[m])
                    if data_mat[k, l, m - 1] != data_mat[k, l, m - 1]:
                        data_mat[k, l, m - 1] = 0.0
                else:
                    # 对应位置进行值编号填入
                    data_mat[k, l, m - 1] = encoding(row[m], value_dict)
    #                 print(value_dict)
    # print(data_mat)
    # print('***********************')

    with open(truth_file_name, errors='ignore', mode='r') as fp2:
        reader = csv.reader(fp2)
        for row in reader:
            if len(row[-1]) == 0:
                continue
            if int(row[-1]) not in obj_dict:
                continue
            # l事实编号
            l = encoding(int(row[-1]), obj_dict)
            for m in range(len(row) - 1):
                if len(row[m]) == 0:
                    continue
                if IsContinuous[m]:
                    truth_mat[l, m] = np.double(row[m])
                    if truth_mat[l, m] != truth_mat[l, m]:
                        print('NaN:(%d,%d)' % (l, m))
                        print(row[m])
                        truth_mat[l, m] = 0.0
                else:
                    # 事实编号填入
                    truth_mat[l, m] = encoding(row[m], value_dict)

    # quit()
    for i in range(L):
        for m in range(M):
            s = np.sum((data_mat[:, i, m] != Invalid_Sign).astype('int'))
            if s == 0:
                valid_mat[i, m] = False
            else:
                # 该事实是否有数据源提供值
                valid_mat[i, m] = True
    for i in range(L):
        for m in range(M):
            if truth_mat[i, m] != truth_mat[i, m]:
                print('NaN:(%d,%d)' % (i, m))
    # quit()
    # print('assadfasdfasdfffffffffffffffffffffffffffffffffffffffffffd')
    return data_mat, truth_mat, valid_mat, K, L, M, value_dict


def load_data_allDiscrete(data_file_name, truth_file_name, IsContinuous):
    M = 0
    L = 0
    source_dict = {}
    obj_dict = {}
    value_dict = {}
    with open(data_file_name, encoding='utf-8', errors='ignore', mode='r') as fp1:
        reader = csv.reader(fp1)
        for row in reader:
            M = len(row) - 2
            if len(row[0]) > 0:
                encoding(int(row[0]), source_dict)
            if len(row[-1]) > 0:
                encoding(int(row[-1]), obj_dict)
    L = len(obj_dict)
    K = len(source_dict)
    print(K, L, M)
    data_mat = np.ones(shape=(K, L, M), dtype=np.int64) * Invalid_Sign
    truth_mat = np.ones(shape=(L, M), dtype=np.int64) * Invalid_Sign
    valid_mat = np.zeros(shape=(L, M), dtype=bool)
    with open(data_file_name, errors='ignore', mode='r') as fp1:
        reader = csv.reader(fp1)
        for row in reader:
            if len(row[0]) == 0:
                continue
            if len(row[-1]) == 0:
                continue
            k, l = encoding(int(row[0]), source_dict), encoding(int(row[-1]), obj_dict)
            for m in range(1, len(row) - 1):
                if len(row[m]) == 0:
                    continue
                if IsContinuous[m - 1]:
                    data_mat[k, l, m - 1] = np.int(row[m])
                    if data_mat[k, l, m - 1] != data_mat[k, l, m - 1]:
                        data_mat[k, l, m - 1] = 0.0
                else:
                    data_mat[k, l, m - 1] = encoding(row[m], value_dict)

    with open(truth_file_name, errors='ignore', mode='r') as fp2:
        reader = csv.reader(fp2)
        for row in reader:
            if len(row[-1]) == 0:
                continue
            if int(row[-1]) not in obj_dict:
                continue
            l = encoding(int(row[-1]), obj_dict)
            for m in range(len(row) - 1):
                if len(row[m]) == 0:
                    continue
                if IsContinuous[m]:
                    truth_mat[l, m] = np.int(row[m])
                    if truth_mat[l, m] != truth_mat[l, m]:
                        print('NaN:(%d,%d)' % (l, m))
                        print(row[m])
                        truth_mat[l, m] = 0.0
                else:
                    truth_mat[l, m] = encoding(row[m], value_dict)
    # quit()
    for i in range(L):
        for m in range(M):
            s = np.sum((data_mat[:, i, m] != Invalid_Sign).astype('int'))
            if s == 0:
                valid_mat[i, m] = False
            else:
                valid_mat[i, m] = True
    for i in range(L):
        for m in range(M):
            if truth_mat[i, m] != truth_mat[i, m]:
                print('NaN:(%d,%d)' % (i, m))
    # quit()
    # print('assadfasdfasdfffffffffffffffffffffffffffffffffffffffffffd')
    return data_mat, truth_mat, valid_mat, K, L, M, value_dict


if __name__ == "__main__":
    data_path = os.path.join('../../data', 'restaurant')
    dataset_IsContinuous = [False, False, False, False, False]
    data_mat_const, ground_truth, valid_mat_const, K, L, M, value_dict = \
        load_data(os.path.join(data_path, 'restaurant_data1.csv'), os.path.join(data_path, 'restaurant_truth1.csv'),
                  dataset_IsContinuous)
