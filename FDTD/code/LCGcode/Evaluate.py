import numpy as np

Invalid_Sign = -1


def evaluate(truth_val, g_truth, valid_mat, IsContinuous, output=False):
    # print(type(truth_val), 'truth_val:', truth_val)
    # np.savetxt("truth_val.csv", truth_val.astype(int), delimiter=",", fmt='%d')
    # fout = open('error_log.txt','w')
    dist = 0.0
    dist_MAD = 0.0
    L, M = g_truth.shape[0], g_truth.shape[1]
    Effective_ground_truth = (g_truth != Invalid_Sign).astype('int') * (valid_mat != False).astype('int')
    n_truth = np.sum(Effective_ground_truth)
    n_truth_categorical = 0
    n_truth_continuous = 0
    totalnum = 0.0
    errors_a = np.zeros(shape=(M,))
    errors = 0
    total_dist = 0.0
    allnum = 0
    for i in range(L):
        vec1 = []
        vec2 = []
        for m in range(M):
            if not valid_mat[i, m]:
                continue
            if (g_truth[i, m] == Invalid_Sign) or (truth_val[i, m] == Invalid_Sign):
                continue
            n_truth_categorical += 1
            if int(round(truth_val[i, m])) != int(round(g_truth[i, m])):
                errors_a[m] += 1
                errors += 1
        if len(vec1) == 0:
            continue
        allnum += 1
    if n_truth_categorical > 0:
        total_error = errors / n_truth_categorical
    else:
        total_error = 0.0
    MNAD, MAD, RMSE = calc_Evaluation(truth_val, g_truth, valid_mat, IsContinuous, output)
    return total_error, MNAD, MAD, RMSE


def calc_Evaluation(truth_val, g_truth, valid_mat, IsContinuous, output):
    dist_MNAD = 0.0
    dist_MAD = 0.0
    dist_RMSE = 0.0
    L, M = g_truth.shape[0], g_truth.shape[1]
    Effective_ground_truth = (g_truth != Invalid_Sign).astype('int') * (valid_mat != False).astype('int')
    n_truth = np.sum(Effective_ground_truth)
    n_truth_categorical = 0
    n_truth_continuous = 0
    totalnum = 0.0
    errors_a = np.zeros(shape=(M,))
    errors = 0
    total_dist = 0.0
    con_all = 0
    for i in range(L):
        vec1 = []
        vec2 = []
        for m in range(M):
            if IsContinuous[m]:
                # Effective_ground_truth[i, m] = 0
                if not valid_mat[i, m]:
                    continue
                if (g_truth[i, m] == Invalid_Sign) or (truth_val[i, m] == Invalid_Sign):
                    continue
                n_truth_continuous += 1
                if g_truth[i, m] != 0:
                    dist_MNAD = dist_MNAD + abs(g_truth[i, m] - truth_val[i, m]) / g_truth[i, m]
                dist_MAD = dist_MAD + abs(g_truth[i, m] - truth_val[i, m])
                dist_RMSE += (g_truth[i, m] - truth_val[i, m]) ** 2
                vec1.append(g_truth[i, m])
                vec2.append(truth_val[i, m])
                con_all += 1
            else:
                if not valid_mat[i, m]:
                    continue
                if (g_truth[i, m] == Invalid_Sign) or (truth_val[i, m] == Invalid_Sign):
                    continue
                n_truth_categorical += 1
                if truth_val[i, m] != g_truth[i, m]:
                    errors_a[m] += 1
                    errors += 1
        if len(vec1) == 0:
            continue
    if con_all > 0:
        MNAD = dist_MNAD / con_all
        MAD = dist_MAD / con_all
        RMSE = np.sqrt(dist_RMSE / con_all)
    else:
        MNAD = MAD = RMSE = 0
    if output:
        print('MNAD:', MNAD)
        print('MAD:', MAD)
        print('RMSE:', RMSE)
    return MNAD, MAD, RMSE
