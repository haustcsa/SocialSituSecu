import os
import datetime
import time
import numpy as np
import sys

sys.path.append('LCGcode')
from LCGcode.Source_reliability import source_sort
from LCGcode.calcCoverage import calcCoverage
from LCGcode.FDTD import FDTD
from LCGcode.load_data import load_data
from LCGcode.DCs_parser import DCs_parser
from LCGcode.Evaluate import evaluate

dataset_Att_list = ['name', 'BUILDING', 'STREET', 'ZIPCODE', 'PHONE']
dataset_IsContinuous = [False, False, False, False, False]
data_path = os.path.join('../data', 'restaurant')
# 数据列表，真值列表，事实是否有值，数据源数量，事实数量，属性数量，属性值列表
data_mat_const, ground_truth, valid_mat_const, K, L, M, value_dict = \
    load_data(os.path.join(data_path, 'restaurant_data.csv'), os.path.join(data_path, 'restaurant_truth.csv'),
              dataset_IsContinuous)

# starttime = datetime.datetime.now()
starttime = time.time()
dcs = DCs_parser(value_dict, dataset_Att_list, dataset_IsContinuous)
source_weight = source_sort(K)
FDTD_start_time = time.process_time()

truth_val, Partition_time, generate_time, iter_time = \
    FDTD(ground_truth, data_mat_const, np.array(source_weight), dcs, dataset_Att_list, dataset_IsContinuous,
        save_block_To_file=False, load_block_From_file=False, max_itor=3, datasetname='restaurant_test')

FDTD_end_time = time.process_time()
# endtime = datetime.datetime.now()
endtime = time.time()
total_error2, MNAD_FDTD, MAD_FDTD, RMSE_FDTD = evaluate(truth_val, ground_truth, valid_mat_const, dataset_IsContinuous)

# coverage = calcCoverage(truth_val, dcs, dataset_Att_list)
print("DC Coverage: ")
# for i in range(len(coverage)):
#     print(coverage[i], dcs[i])

print('source_weight', source_weight)
print('FDTD error rate:' + str(total_error2))
print('FDTD Accuracy:' + str(100 - total_error2 * 100))
print('FDTD Iter Time(sec):' + str(iter_time))

fout = open('../results/FDTD Results.txt', 'w')
fout.write('FDTD Accuracy:' + str(100 - total_error2 * 100) + '\n')
fout.write('FDTD error rate:' + str(total_error2) + '\n')
fout.write('FDTD Iter Time(sec):' + str(iter_time) + '\n')
fout.write('FDTD Partition Time(sec):' + str(Partition_time) + '\n')
fout.write('FDTD Generate Time(sec):' + str(generate_time) + '\n')
fout.write('FDTD Iter Time(sec):' + str(iter_time) + '\n')
fout.write('FDTD NoPartition Time(sec):' + str(generate_time + iter_time) + '\n')
fout.write('FDTD Total CPUTime(sec):' + str(FDTD_end_time - FDTD_start_time) + '\n')
fout.write('FDTD Total RunTime(sec):' + str(endtime - starttime) + '\n')
fout.close()
