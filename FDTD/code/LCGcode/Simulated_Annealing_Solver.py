import random
import time
import numpy as np

def annealing_solver(g_truth, claim_decode_map, x0, E0, generateX, gen_func_args, T0 = 50, alpha = 0.9, Tmin = 0.01, max_iter = 5):
#def annealing_solver(x0, generateX, gen_func_args, E_func, E_func_args, T0 = 100, alpha = 0.96, Tmin = 0.001, max_iter = 10, print_info = False):
    T = T0
    x = x0
    #print(len(x0))
    #Ex = E_func(x, E_func_args)
    Ex = E0
    error_rate_list = []
    H_func_list = []
    '''
    H_func_list.append(Ex)
    error_num = 0.0
    for i in range(len(x)):
        for claim in claim_decode_map[i]:
            if g_truth[claim[0], claim[1]] != x[i]:
                error_num += 1
    error_rate_list.append(error_num / len(x))
    '''
    while T > Tmin:
        for i in range(max_iter):
            xnew, Enew = generateX(x, Ex, gen_func_args)
            #xnew = generateX(x, gen_func_args)
            #Enew = E_func(xnew, E_func_args) 
            delta_E = Enew - Ex
            #print(delta_E)
            if delta_E < 0:
                x = xnew #accept
                Ex = Enew
            else:
                poss = np.exp(- delta_E / T)
                if random.uniform(0, 1) <= poss:  #accept
                    x = xnew
                    Ex = Enew
            '''
            if Ex > 280:
                continue
            H_func_list.append(Ex)
            error_num = 0.0
            for i in range(len(x)):
                for claim in claim_decode_map[i]:
                    if g_truth[claim[0], claim[1]] != x[i]:
                        error_num += 1
            #error_rate_list.append(round(error_num / len(x), 2))
            error_rate_list.append(error_num / len(x))
            '''
        T = alpha * T
        
    return x, error_rate_list, H_func_list