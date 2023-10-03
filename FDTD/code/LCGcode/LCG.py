from DCs_parser import *
from DCs_sorted import *
from union_find_sets import *
import numpy as np
from Stack import *
from Simulated_Annealing_Solver import *
import pickle
import itertools
import cvxopt
import time
import matplotlib.pyplot as plt
from DCTree import *
Invalid_Sign = -1

def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    P = P.astype(np.double)
    q = q.astype(np.double)
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        G = G.astype(np.double)
        h = h.astype(np.double)
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            A = A.astype(np.double)
            b = b.astype(np.double)
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],))

def generate_value_dict(data_mat, source_weight, i, j):
    value_list = data_mat[:,i,j]
    if source_weight.ndim == 1:
        weight_sum = np.sum(source_weight)
    else:
        assert source_weight.ndim == 2
        weight_sum = np.sum(source_weight[:, j])
    value_dict = dict()
    for i in range(len(value_list)):
        value = value_list[i]
        if value == Invalid_Sign:
            continue
        if value not in value_dict:
            value_dict[value] = 0.0
        if weight_sum == 0:
            continue
        if source_weight.ndim == 1:
            if weight_sum > 0:
                value_dict[value] += source_weight[i]/weight_sum
        else:
            assert source_weight.ndim == 2
            if weight_sum > 0:
                value_dict[value] += source_weight[i, j]/weight_sum
    return value_dict

def generate_conflict_blocks_old(truth_mat, dc,Att_list):
    L, M = truth_mat.shape[0],truth_mat.shape[1]
    blocks_set = union_find_sets(L)
    for i in range(L):
        for j in range(i,L):
            if i == j:
                continue
            claimA, claimB = truth_mat[i,:], truth_mat[j,:]
            if dc.Is_Conflicting(Att_list, claimA, claimB):
                blocks_set.union(i,j)
    blocks_list = blocks_set.generate_sets()
    return [s for s in blocks_list if len(s) > 1]

def NeedMerged(block1, block2, data_mat, dc, Att_list):
    for t1 in block1:
        for t2 in block2:
            if dc.Is_Potentially_Conflicting(t1, t2, data_mat, Att_list):
                return True
    return False

def NeedAdded(block, data_mat, dc, Att_list):
    for t1 in block:
        for t2 in block:
            if t1 == t2:
                continue
            if dc.Is_Potentially_Conflicting(t1, t2, data_mat, Att_list):
                return True
    return False


def generate_conflict_blocks(data_mat, dcs, Att_list):
    tree = DCTree()
    dc_list = list(dcs)
    dcsComparator = ClauseComparator(dc_list,data_mat,Att_list)
    for i in range(len(dc_list)):
        if dc_list[i].Is_Single_Entity():
            continue
        dc_list[i].clause_list = list(set(dc_list[i].clause_list))
        dc_list[i].sort_clauses(dcsComparator)
        dc = dc_list[i]
        tree.add_DC(dc,i)
    worker = DCTreeWorker(tree,data_mat,Att_list)
    blocks_list, blocks_dc_index_list = worker.work()
    for blockId in range(len(blocks_list)):
        for x in blocks_list[blockId]: # 枚举块内点
            for i in range(len(dc_list)):
                if dc_list[i].Is_Single_Entity(): # 枚举single entity dc
                    dc = dc_list[i]
                    ans = False
                    for cla in dc.clause_list:
                        if not cla.MayBeTrue(x, None, data_mat, Att_list):
                            ans = True
                            break
                    if not ans:
                        blocks_dc_index_list[blockId].add(i)
    blocks_dc_list = []
    for blockIndex in range(len(blocks_list)):
        s = set()
        for j in blocks_dc_index_list[blockIndex]:
            s.add(dc_list[j])
        blocks_dc_list.append(s)
    blocks_len = []
    for i in range(len(blocks_list)):
        if len(blocks_list[i]) > 1:
            blocks_len.append(len(blocks_list[i]))
    if len(blocks_len) == 0:
        print("No Edge")
    else:
        print('Max block: ',np.max(blocks_len))
        print('len: ',len(blocks_len))
        print('means: ',np.mean(blocks_len))
        print('median: ',np.median(blocks_len))
    return blocks_list, blocks_dc_list



def calc_probability(weight_sum, value_dictA, value_dictB, op):
    if isinstance(value_dictB,dict):
        value_listA = list(value_dictA.items())
        value_listB = list(value_dictB.items())
        all_case = 0.0
        for i in range(len(value_listA)):
            for j in range(len(value_listB)):
                all_case += value_listA[i][1] * value_listB[j][1]
        zi = 0.0
        if op == '!=':
            for i in range(len(value_listA)):
                for j in range(len(value_listB)):
                    if value_listA[i][0] != value_listB[j][0]:
                        zi += value_listA[i][1] * value_listB[j][1]
        # elif op == '>':  #A>B
        #     for i in range(len(value_listA)):
        #         for j in range(len(value_listB)):
        #             if value_listA[i][0] > value_listB[j][0]:
        #                 zi += value_listA[i][1] * value_listB[j][1]
        # elif op == '<':
        #     for i in range(len(value_listA)):
        #         for j in range(len(value_listB)):
        #             if value_listA[i][0] < value_listB[j][0]:
        #                 zi += value_listA[i][1] * value_listB[j][1]
        elif op == '=':
            for i in range(len(value_listA)):
                for j in range(len(value_listB)):
                    if value_listA[i][0] == value_listB[j][0]:
                        zi += value_listA[i][1] * value_listB[j][1]
        if all_case == 0.0:
            return 0.0
        return zi/all_case
    else:
        value_listA = list(value_dictA.items())
        value_listB = value_dictB
        zi = 0.0
        all_case = sum([x[1] for x in value_listA[:]])
        if op == '!=':
            for i in range(len(value_listA)):
                if value_listA[i][0] != value_listB:
                    zi += value_listA[i][1]
        # elif op == '>':  #A>B
        #     for i in range(len(value_listA)):
        #         if value_listA[i][0] > value_listB:
        #             zi += value_listA[i][1]
        # elif op == '<':
        #     for i in range(len(value_listA)):
        #         if value_listA[i][0] < value_listB:
        #             zi += value_listA[i][1]
        elif op == '=':
            for i in range(len(value_listA)):
                if value_listA[i][0] == value_listB:
                    zi += value_listA[i][1]
        return zi/all_case

def calc_probability_new(weight_sum, value_dictA, value_dictB, op):  #can be improved
    if isinstance(value_dictB,dict):
        value_listA = list(value_dictA.items())
        value_listB = list(value_dictB.items())
        zi = 0.0
        if op == '!=':
            for i in range(len(value_listA)):
                for j in range(len(value_listB)):
                    if value_listA[i][0] != value_listB[j][0]:
                        zi += value_listA[i][1] * value_listB[j][1]
        # elif op == '>':  #A>B
        #     for i in range(len(value_listA)):
        #         for j in range(len(value_listB)):
        #             if value_listA[i][0] > value_listB[j][0]:
        #                 zi += value_listA[i][1] * value_listB[j][1]
        # elif op == '<':
        #     for i in range(len(value_listA)):
        #         for j in range(len(value_listB)):
        #             if value_listA[i][0] < value_listB[j][0]:
        #                 zi += value_listA[i][1] * value_listB[j][1]
        elif op == '=':
            for i in range(len(value_listA)):
                for j in range(len(value_listB)):
                    if value_listA[i][0] == value_listB[j][0]:
                        zi += value_listA[i][1] * value_listB[j][1]
        return zi / weight_sum
    else:
        value_listA = list(value_dictA.items())
        value_listB = value_dictB
        zi = 0.0
        if op == '!=':
            for i in range(len(value_listA)):
                if value_listA[i][0] != value_listB:
                    zi += value_listA[i][1]
        # elif op == '>':  #A>B
        #     for i in range(len(value_listA)):
        #         if value_listA[i][0] > value_listB:
        #             zi += value_listA[i][1]
        # elif op == '<':
        #     for i in range(len(value_listA)):
        #         if value_listA[i][0] < value_listB:
        #             zi += value_listA[i][1]
        elif op == '=':
            for i in range(len(value_listA)):
                if value_listA[i][0] == value_listB:
                    zi += value_listA[i][1]
        return zi / (weight_sum ** 2)

def generate_constraints(entities, cla, Att_list):
    new_constraint_op = ''
    if cla.op == '!=':
        new_constraint_op = '='
    elif cla.op == '=':
        new_constraint_op = '!='
    # elif cla.op == '>':
    #     new_constraint_op = '<='
    # elif cla.op == '<':
    #     new_constraint_op = '>='
    # elif cla.op == '>=':
    #     new_constraint_op = '<'
    # elif cla.op == '<=':
    #     new_constraint_op = '>'
    atts = cla.translate(Att_list)
    if atts == None:
        return None
    if len(entities) == 2:
        return (entities[0],atts[0],entities[1],atts[1],new_constraint_op)
    else:
        if len(atts) == 2:
            return (entities[0],atts[0],entities[0],atts[1],new_constraint_op)
        else:
            return (entities[0],atts[0],cla.constant,new_constraint_op)
    return None

def judge(A,B,op):
    this_value, other_value = A, B
    if op == '=':
        return this_value == other_value
    elif op == '!=':
        return this_value != other_value
    # elif op == '<':
    #     return this_value < other_value
    # elif op == '>':
    #     return this_value > other_value
    # elif op == '>=':
    #     return this_value >= other_value
    # elif op == '<=':
    #     return this_value <= other_value
    return True

def judge_by_op(A,B,this_index_claim,this_value,other_value,op):
    if B == this_index_claim:
        this_value, other_value = other_value, this_value
    if op == '=':
        return this_value == other_value
    elif op == '!=':
        return this_value != other_value
    # elif op == '<':
    #     return this_value < other_value
    # elif op == '>':
    #     return this_value > other_value
    # elif op == '>=':
    #     return this_value >= other_value
    # elif op == '<=':
    #     return this_value <= other_value
    return True

def Is_Conflicting_In_Handled_Set(LC_dict_No_Constant, ClaimsInBlock_list, Handled_claim_value_dict, this_index, value):
    if (ClaimsInBlock_list[this_index][0], ClaimsInBlock_list[this_index][1]) in LC_dict_No_Constant:
        lc_list = LC_dict_No_Constant[(ClaimsInBlock_list[this_index][0], ClaimsInBlock_list[this_index][1])]
    else:
        lc_list = []
    this_index_claim = (ClaimsInBlock_list[this_index][0], ClaimsInBlock_list[this_index][1])
    legal = True
    for lc in lc_list:
        A = (lc[0],lc[1])
        B = (lc[2],lc[3])
        other_value = None
        if A == this_index_claim:
            if B not in Handled_claim_value_dict:
                continue
            other_value = Handled_claim_value_dict[B]
        elif B == this_index_claim:
            if A not in Handled_claim_value_dict:
                continue
            other_value = Handled_claim_value_dict[A]
        if other_value == None:
            continue
        if not judge_by_op(A,B,this_index_claim,value,other_value,lc[-1]):
            legal = False
            break
    return not legal

def get_valid_values_set(data_mat, LC):
    op = LC[-1]
    value_set = set()
    if len(LC) == 4:
        claim = (LC[0],LC[1])
        value_list = data_mat[:,LC[0],LC[1]]
        value_list = value_list[value_list != Invalid_Sign]
        for value in value_list:
            if judge_by_op(claim,(-1,-1),claim,value,LC[3],op):
                value_set.add(value)
        return value_set
    else:
        claimA = (LC[0],LC[1])
        claimB = (LC[2],LC[3])
        value_listA = data_mat[:,LC[0],LC[1]]
        value_listA = value_listA[value_listA != Invalid_Sign]
        value_listB = data_mat[:,LC[2],LC[3]]
        value_listB = value_listB[value_listB != Invalid_Sign]
        for valueA in value_listA:
            for valueB in value_listB:
                if judge_by_op(claimA,claimB,claimA,valueA,valueB,op):
                    value_set.add((valueA,valueB))
        return value_set

def get_value_list(tuple, att, data_mat):
    value_list = data_mat[:,tuple,att]
    return value_list[value_list != Invalid_Sign]

def get_available_values(data_mat, LC):
    value_set = set()
    if len(LC) == 4:
        if LC[-1] == '=':
            value_set.add(LC[-2])
            return list(value_set)
        value_list = get_value_list(LC[0], LC[1], data_mat)
        op = LC[-1]
        for value in value_list:
            if judge(value, LC[-2], op):
                value_set.add(value)
        return list(value_set)
    else:
        value_listA = get_value_list(LC[0], LC[1], data_mat)
        value_listB = get_value_list(LC[2], LC[3], data_mat)
        values_all = set(value_listA) | set(value_listB)
        op = LC[-1]
        if op == '=':
            for value in values_all:
                value_set.add((value, value))
            return list(value_set)
        #all_value_combs = itertools.product(value_list_all, repeat=2)
        for valueA in values_all:
            for valueB in values_all:
                if judge(valueA, valueB, op):
                    value_set.add((valueA, valueB))
        return list(value_set)

def Solve_Conflict_On_Categorical_Data_new(data_mat, source_weight, truth_mat, ClaimsInBlock_set, LC_set, IsContinuous):
    if len(ClaimsInBlock_set) == 0:
        return
    LC_list = list(LC_set)
    ClaimsInBlock_Categorical_Data_set = set()
    for claim in ClaimsInBlock_set:
        if not IsContinuous[claim[1]]:
            ClaimsInBlock_Categorical_Data_set.add(claim)
    ClaimsInBlock_list = list(ClaimsInBlock_Categorical_Data_set)
    dfs_stack = Stack()
    dfs_stack.push((-1,set()))
    value_map = dict()
    value_dict_map = dict()
    visited_map = dict()
    best_answer = None
    best_weight = -1e-10
    LC_available_values_list = []
    for i in range(len(LC_list)):
        LC_available_values_list.append(get_available_values(data_mat, LC_list[i]))
    while not dfs_stack.isEmpty():
        now, added_new_value_cells = dfs_stack.peek()
        succ = now  + 1
        #print(succ)
        if succ >= len(LC_list):
            total_weight = 0.0
            for cell,value in value_map.items():
                if cell not in value_dict_map:
                    value_dict_map[cell] = generate_value_dict(data_mat, source_weight, cell[0], cell[1])
                value_dict = value_dict_map[cell]
                if value in value_dict:
                    total_weight += value_dict[value]
            if total_weight > best_weight:
                best_weight = total_weight
                best_answer = dict()
                for cell,value in value_map.items():
                    best_answer[cell] = value
            for cell in added_new_value_cells:
                if cell in value_map:
                    value_map.pop(cell)
            if succ in visited_map:
                visited_map.pop(succ)
            dfs_stack.pop()
            #print('to the last!!!!!!!!!!!!!!!!!')
            continue
        new_value_cells = set()
        LC = LC_list[succ]
        LC_available_values = LC_available_values_list[succ]
        available_values_ind = 0
        if succ in visited_map:
            #print('succ in visited_map')
            available_values_ind = visited_map[succ] + 1  #别忘了更新visited_map[succ]
        has_solved = 0
        while available_values_ind < len(LC_available_values):
            if len(LC) == 4:
                cell = (LC[0],LC[1])
                if cell in value_map:
                    has_solved = 1
                    if value_map[cell] not in LC_available_values:
                        has_solved = 2
                    break
                value_map[cell] = LC_available_values[available_values_ind]
                new_value_cells.add(cell)
                dfs_stack.push((succ, new_value_cells))
                break
            else:
                cellA = (LC[0],LC[1])
                cellB = (LC[2],LC[3])
                if (cellA in value_map) and (cellB in value_map):
                    if (value_map[cellA],value_map[cellB]) not in LC_available_values:
                        has_solved = 2
                    else:
                        has_solved = 1
                    break
                valueA = None
                ValueB = None
                if cellA in value_map:
                    if value_map[cellA] == LC_available_values[available_values_ind][0]:
                        value_map[cellB] = LC_available_values[available_values_ind][1]
                        new_value_cells.add(cellB)
                        dfs_stack.push((succ, new_value_cells))
                        break
                if cellB in value_map:
                    if value_map[cellB] == LC_available_values[available_values_ind][1]:
                        value_map[cellA] = LC_available_values[available_values_ind][0]
                        new_value_cells.add(cellA)
                        dfs_stack.push((succ, new_value_cells))
                        break
                new_value_cells.add(cellA)
                new_value_cells.add(cellB)
                value_map[cellA] = LC_available_values[available_values_ind][0]
                value_map[cellB] = LC_available_values[available_values_ind][1]
                dfs_stack.push((succ, new_value_cells))
                break
            available_values_ind += 1
        if has_solved == 1:
            dfs_stack.push((succ, new_value_cells))
            visited_map[succ] = len(LC_available_values) - 1
            continue
        if (available_values_ind == len(LC_available_values)) or (has_solved == 2):  #未找到合适的后继
            for cell in added_new_value_cells:
                if cell in value_map:
                    value_map.pop(cell)
            #print(available_values_ind == len(LC_available_values))
            if succ in visited_map:
                #print('pop succ ')
                visited_map.pop(succ)
            dfs_stack.pop()
        else:
            visited_map[succ] = available_values_ind

    if best_answer != None:
        for claim,value in best_answer.items():
            entity = claim[0]
            att = claim[1]
            truth_mat[entity,att] = value
            #print('I found!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

def sort_att_LC_list(att_LC_list):
    tuple_count_dict = dict()
    for LC in att_LC_list:
        if LC[0] not in tuple_count_dict:
            tuple_count_dict[LC[0]] = 0
        tuple_count_dict[LC[0]] += 1
        if len(LC) == 5:
            if LC[2] not in tuple_count_dict:
                tuple_count_dict[LC[2]] = 0
            tuple_count_dict[LC[2]] += 1
    sorted_tuple_count_list = sorted(tuple_count_dict.items(),key = lambda x : x[1], reverse = True)
    all_LC_set = set(att_LC_list)
    new_LC_list = []
    for tuple_num, times in sorted_tuple_count_list:
        for LC in att_LC_list:
            if (LC[1] == tuple_num) or ((len(LC) == 5) and (LC[2] == tuple_num)):
                if LC in all_LC_set:
                    all_LC_set.remove(LC)
                    new_LC_list.append(LC)
        if len(all_LC_set) == 0:
            break
    if len(new_LC_list) != len(att_LC_list):
        print('error!!!!!!!!!!!!!!!!')
    return new_LC_list

def sort_LC_list(LC_list):
    ind = 0
    LC_dict = dict()
    for LC in LC_list:
        if LC[1] not in LC_dict:
            LC_dict[LC[1]] = []
        LC_dict[LC[1]].append(LC)
    LC_items = []
    for att,att_LC_list in LC_dict.items():
        LC_items.append((att,sort_att_LC_list(att_LC_list)))
    LC_items = sorted(LC_items, key = lambda x : x[0], reverse = False)
    ans = []
    for att,att_LC_list in LC_items:
        for LC in att_LC_list:
            ans.append(LC)
    return ans

def LC_encode(LC_list):
    claim_dict = dict()
    decode_dict = dict()
    for LC in LC_list:
        if (LC[0],LC[1]) not in claim_dict:
            decode_dict[len(claim_dict)] = (LC[0],LC[1])
            claim_dict[(LC[0],LC[1])] = len(claim_dict)
        if len(LC) == 5:
            if (LC[2],LC[3]) not in claim_dict:
                decode_dict[len(claim_dict)] = (LC[2],LC[3])
                claim_dict[(LC[2],LC[3])] = len(claim_dict)
    union_set = union_find_sets(len(claim_dict))
    for LC in LC_list:
        if (len(LC) == 5) and (LC[-1] == '='):
            union_set.union(claim_dict[(LC[0],LC[1])],claim_dict[(LC[2],LC[3])])
    claim_blocks_sets_list = list(union_set.generate_sets())
    LC_decode_map = dict()
    LC_encode_map = dict()
    num_set = set()
    for i in range(len(claim_blocks_sets_list)):
        claim_block = claim_blocks_sets_list[i]
        LC_decode_map[i] = []
        num_set.add(i)
        for claim_num in claim_block:
            claim = decode_dict[claim_num]
            LC_decode_map[i].append(claim)
            LC_encode_map[claim] = i
    encoded_LC_list = []

    for LC in LC_list:
        if len(LC) == 4:
            encoded_LC_list.append((LC_encode_map[(LC[0],LC[1])],LC[2],LC[3]))
            if LC_encode_map[(LC[0],LC[1])] in num_set:
                num_set.remove(LC_encode_map[(LC[0],LC[1])])
        else:
            if LC_encode_map[(LC[0],LC[1])] == LC_encode_map[(LC[2],LC[3])]:
                continue
            encoded_LC_list.append((LC_encode_map[(LC[0],LC[1])],LC_encode_map[(LC[2],LC[3])],1017,LC[4]))
            if LC_encode_map[(LC[0],LC[1])] in num_set:
                num_set.remove(LC_encode_map[(LC[0],LC[1])])
            if LC_encode_map[(LC[2],LC[3])] in num_set:
                num_set.remove(LC_encode_map[(LC[2],LC[3])])

    for num in num_set:
        encoded_LC_list.append((num, num, 1017, '='))

    return encoded_LC_list, LC_encode_map, LC_decode_map

def get_available_values_with_encoding(value_dict_map, LC):
    value_set = set()
    if len(LC) == 3:
        ans = []
        if LC[-1] == '=':
            ans.append(LC[-2])
            return ans
        value_list = sorted(value_dict_map[LC[0]].items(), key = lambda x : x[1], reverse = True)
        op = LC[-1]
        for value,weight in value_list:
            if judge(value, LC[-2], op):
                ans.append(value)
        return ans
    else:
        op = LC[-1]
        if op == '=':
            value_map = dict()
            for value,weight in value_dict_map[LC[0]].items():
                if (value, value) not in value_map:
                    value_map[(value, value)] = 0.0
                value_map[(value, value)] += weight
            for value,weight in value_dict_map[LC[1]].items():
                if (value, value) not in value_map:
                    value_map[(value, value)] = 0.0
                value_map[(value, value)] += weight
            value_list = sorted(value_map.items(), key = lambda x : x[1], reverse = True)
            return [v for v,k in value_list]
        value_map = dict()
        for valueA,weightA in value_dict_map[LC[0]].items():
            for valueB,weightB in value_dict_map[LC[1]].items():
                if not judge(valueA, valueB, op):
                    continue
                if (valueA, valueB) not in value_map:
                    value_map[(valueA, valueB)] = 0.0
                value_map[(valueA, valueB)] += (weightA + weightB)
        value_list = sorted(value_map.items(), key = lambda x : x[1], reverse = True)
        return [v for v,k in value_list]


def generateX(x0, E_old, args):
    value_list_map = args[0]
    value_dict_map = args[1]
    value_sum_map = args[2]
    encoded_LC_list = args[3]
    involved_LC_map = args[4]
    E_new = E_old
    claim_num = random.randint(0, len(value_list_map) - 1)
    ind = random.randint(0, len(value_list_map[claim_num]) - 1)
    x_new = x0.copy()
    x_new[claim_num] = value_list_map[claim_num][ind]
    E_new += value_dict_map[claim_num][x0[claim_num]]
    E_new -= value_dict_map[claim_num][x_new[claim_num]]
    for LC in involved_LC_map[claim_num]:

        valueA = x0[LC[0]]
        if len(LC) == 3:
            if not judge(valueA,LC[1],LC[-1]):
                E_new -= value_sum_map[LC[0]] * 2
        else:
            valueB = x0[LC[1]]
            if not judge(valueA,valueB,LC[-1]):
                E_new -= value_sum_map[LC[0]]
                E_new -= value_sum_map[LC[1]]

        valueA = x_new[LC[0]]
        if len(LC) == 3:
            if not judge(valueA,LC[1],LC[-1]):
                E_new += value_sum_map[LC[0]] * 2
        else:
            valueB = x_new[LC[1]]
            if not judge(valueA,valueB,LC[-1]):
                E_new += value_sum_map[LC[0]]
                E_new += value_sum_map[LC[1]]
    return x_new, E_new

def E_evaluate_func(x, args):
    value_dict_map = args[0]
    value_sum_map = args[1]
    encoded_LC_list = args[2]
    ans = 0
    penalty = 0
    for i in range(len(value_dict_map)):
        ans += value_sum_map[i]
        ans -= value_dict_map[i][x[i]]
    penalty = ans
    for LC in encoded_LC_list:
        valueA = x[LC[0]]
        if len(LC) == 3:
            if not judge(valueA,LC[1],LC[-1]):
                ans += value_sum_map[LC[0]] * 2
        else:
            valueB = x[LC[1]]
            if not judge(valueA,valueB,LC[-1]):
                ans += value_sum_map[LC[0]]
                ans += value_sum_map[LC[1]]
    penalty = ans - penalty
    return ans, penalty

def Solve_Conflict_On_Categorical_Data_with_Simulated_Annealing(g_truth, data_mat, source_weight, truth_mat, ClaimsInBlock_set, LC_set, IsContinuous):
    LC_list = []
    for LC in LC_set:
        if len(LC) == 4:
            if not IsContinuous[LC[1]]:
                LC_list.append(LC)
        else:
            if (not IsContinuous[LC[1]]) and (not IsContinuous[LC[3]]):
                LC_list.append(LC)
    LC_block = LC_list
    value_map = dict()
    value_dict_map = dict()
    value_list_map = dict()
    value_sum_map = dict()
    involved_LC_map = dict()
    encoded_LC_list, claim_encode_map, claim_decode_map = LC_encode(LC_block)
    if len(encoded_LC_list) == 0:
        best_answer = 0
        max_weight = -1
        for var_num, real_claims in claim_decode_map.items():
            var_value_dict = dict()
            for claim in real_claims:
                value_dict = generate_value_dict(data_mat, source_weight, claim[0], claim[1])
                for value,weight in value_dict.items():
                    if value not in var_value_dict:
                        var_value_dict[value] = 0.0
                    var_value_dict[value] += weight
            for value ,w in var_value_dict.items():
                if w > max_weight:
                    max_weight = w
                    best_answer = value
            for claim in real_claims:
                truth_mat[claim[0], claim[1]] = best_answer
        return
    for LC in encoded_LC_list:
        if LC[0] not in involved_LC_map:
            involved_LC_map[LC[0]] = []
        involved_LC_map[LC[0]].append(LC)
        if len(LC) == 4:
            if LC[1] not in involved_LC_map:
                involved_LC_map[LC[1]] = []
            involved_LC_map[LC[1]].append(LC)
    for var_num, real_claims in claim_decode_map.items():
        var_value_dict = dict()
        for claim in real_claims:
            value_dict = generate_value_dict(data_mat, source_weight, claim[0], claim[1])
            for value,weight in value_dict.items():
                if value not in var_value_dict:
                    var_value_dict[value] = 0.0
                var_value_dict[value] += weight
        value_dict_map[var_num] = var_value_dict
        value_sum_map[var_num] = np.sum(list(var_value_dict.values()))
        value_list_map[var_num] = list(var_value_dict.keys())
    LC_available_values_list = []
    for i in range(len(encoded_LC_list)):
        LC_available_values_list.append(get_available_values_with_encoding(value_dict_map, encoded_LC_list[i]))
    x0 = []
    for i in range(len(claim_decode_map)):
        ind = random.randint(0, len(value_list_map[i]) - 1)
        x0.append(value_list_map[i][ind])
    E_func_args = (value_dict_map, value_sum_map, encoded_LC_list)
    gen_func_args = (value_list_map, value_dict_map, value_sum_map, encoded_LC_list, involved_LC_map)
    E0, p0 = E_evaluate_func(x0, E_func_args)
    x_ans, error_rate_list, H_func_list = annealing_solver(g_truth, claim_decode_map, x0, E0, generateX, gen_func_args)
    for i in range(len(x_ans)):
        for claim in claim_decode_map[i]:
            truth_mat[claim[0], claim[1]] = x_ans[i]


def get_value_weight(data_mat, source_weight, cell, value):
    if source_weight.ndim == 1:
        return np.sum(source_weight[data_mat[:,cell[0],cell[1]] == value])
    else:
        return np.sum(source_weight[data_mat[:,cell[0],cell[1]] == value][cell[1]])

def generatePq(data_mat, source_weight, ClaimNum_dict, ClaimsInBlock_list):
    var_num = len(ClaimsInBlock_list)
    P = np.zeros(shape=(var_num, var_num), dtype=np.double)
    q = np.zeros(shape=(var_num,), dtype=np.double)
    for i in range(var_num):
        l, p = ClaimsInBlock_list[ClaimNum_dict[ClaimsInBlock_list[i]]]   ####can be improved
        valid_array = (data_mat[:, l, p] != Invalid_Sign)
        if source_weight.ndim == 1:
            valid_source = source_weight[valid_array]
        else:
            assert source_weight.ndim == 2
            valid_source = source_weight[valid_array, p]
        valid_claim_value = data_mat[:, l, p][valid_array]
        P[i, i] = 2.0 * np.sum(valid_source)
        q[i] = - 2.0 * np.sum(valid_source * valid_claim_value)
    return P, q.T

def generateGh(data_mat, source_weight, ClaimNum_dict, ClaimsInBlock_list, other_lc, constant_lc):
    var_num = len(ClaimsInBlock_list)
    LCs_set = set(other_lc.values()) | set(constant_lc.values())
    G = None
    h = None
    for LC in LCs_set:
        if len(LC) == 5:
            x_A = ClaimNum_dict[(LC[0],LC[1])]
            x_B = ClaimNum_dict[(LC[2],LC[3])]
            op = LC[-1]
            new_row = np.zeros(shape=(1,var_num), dtype=np.double)
            if op == '<' or op == '<=':
                new_row[0][x_A] = 1
                new_row[0][x_B] = -1
            elif op == '>' or op == '>=':
                new_row[0][x_A] = -1
                new_row[0][x_B] = 1
            if G is None:
                G = new_row
            else:
                G = np.row_stack((G, new_row))
            if h is None:
                h = np.array([0], dtype=np.double)
            else:
                h = np.row_stack((h, np.array([0], dtype=np.double)))
        else:
            x_A = ClaimNum_dict[(LC[0],LC[1])]
            C = LC[2]
            op = LC[-1]
            new_row = np.zeros(shape=(1,var_num), dtype=np.double)
            if op == '<' or op == '<=':
                new_row[0][x_A] = 1
                if h is None:
                    h = np.array([C], dtype=np.double)
                else:
                    h = np.row_stack((h, np.array([C], dtype=np.double)))
            elif op == '>' or op == '>=':
                new_row[0][x_A] = -1
                if h is None:
                    h = np.array([-C], dtype=np.double)
                else:
                    h = np.row_stack((h, np.array([-C], dtype=np.double)))
            else:
                if h is None:
                    h = np.array([0], dtype=np.double)
                else:
                    h = np.row_stack((h, np.array([0], dtype=np.double)))
            if G is None:
                G = new_row
            else:
                G = np.row_stack((G, new_row))
    return G, h

def generateAb(data_mat, source_weight, ClaimNum_dict, ClaimsInBlock_list, other_lc, constant_lc):
    var_num = len(ClaimsInBlock_list)
    LCs_set = set(other_lc.values()) | set(constant_lc.values())
    A = None
    b = None
    for LC in LCs_set:
        op = LC[-1]
        A_new_row = np.zeros(shape=(1,var_num), dtype=np.double)
        b_new_row = np.zeros(shape=(1,1), dtype=np.double)
        if op == '=':
            if len(LC) == 4:
                x_A = ClaimNum_dict[(LC[0],LC[1])]
                A_new_row[0][x_A] = 1
                b_new_row[0][0] = LC[2]
            else:
                x_A = ClaimNum_dict[(LC[0],LC[1])]
                x_B = ClaimNum_dict[(LC[2],LC[3])]
                A_new_row[0][x_A] = 1
                A_new_row[0][x_B] = -1
                b_new_row[0][0] = 0
            if A is None:
                A = A_new_row
            else:
                A = np.row_stack((A, A_new_row))
            if b is None:
                b = b_new_row
            else:
                b = np.row_stack((b, b_new_row))
    return A, b

def Solve_Conflict_On_Continuous_Data(data_mat, source_weight, truth_mat, ClaimsInBlock_set, LC_set, IsContinuous):
    if len(ClaimsInBlock_set) == 0:
        return
    constant_lc = dict()
    other_lc = dict()
    # for LC in LC_set:
    #     if len(LC) == 4:
    #         if IsContinuous[LC[1]]:
    #             constant_lc[(LC[0],LC[1])] = LC
    #     else:
    #         if IsContinuous[LC[1]]:
    #             other_lc[(LC[0],LC[1])] = LC
    #         if IsContinuous[LC[3]]:
    #             other_lc[(LC[2],LC[3])] = LC
    ClaimsInBlock_Continuous_Data_set = set()
    # for claim in ClaimsInBlock_set:
    #     if IsContinuous[claim[1]]:
    #         ClaimsInBlock_Continuous_Data_set.add(claim)
    ClaimsInBlock_list = list(ClaimsInBlock_Continuous_Data_set)
    ClaimNum_dict = dict()
    for i in range(len(ClaimsInBlock_list)):
        ClaimNum_dict[ClaimsInBlock_list[i]] = i
    P, q = generatePq(data_mat, source_weight, ClaimNum_dict, ClaimsInBlock_list)
    G, h = generateGh(data_mat, source_weight, ClaimNum_dict, ClaimsInBlock_list, other_lc, constant_lc)
    A, b = generateAb(data_mat, source_weight, ClaimNum_dict, ClaimsInBlock_list, other_lc, constant_lc)
    X_mat = cvxopt_solve_qp(P, q, G, h, A, b)
    for i in range(len(X_mat)):
        truth_mat[ClaimsInBlock_list[i][0],ClaimsInBlock_list[i][1]] = X_mat[i]

def Print_LCs(LCs, Att_list):
    for LC in LCs:
        if len(LC) == 4:
            print('t%d.%s%s%d'%(LC[0],Att_list[LC[1]],LC[-1],LC[2]))
        else:
            print('t%d.%s%st%d.%s'%(LC[0],Att_list[LC[1]],LC[-1],LC[2],Att_list[LC[3]]))

def LCG(data_mat, source_weight, dcs, Att_list, IsContinuous, save_block_To_file = False, load_block_From_file = False, datasetname = '', only_generate_conflict_blocks = False):
    Partition_time = 0.0
    if load_block_From_file:
        with open('blocks_list'+datasetname+'.dat', 'rb') as f:
            blocks_list = pickle.load(f)
        with open('blocks_dc_list'+datasetname+'.dat', 'rb') as f:
            blocks_dc_list = pickle.load(f)
    else:
        print('generating conflicting blocks...')
        Partition_start_time = time.clock()
        blocks_list, blocks_dc_list = generate_conflict_blocks(data_mat, dcs, Att_list)
        Partition_end_time = time.clock()
        Partition_time = Partition_end_time - Partition_start_time
    if save_block_To_file:
        with open('blocks_list'+datasetname+'.dat', 'wb') as f:
            pickle.dump(blocks_list, f)
        with open('blocks_dc_list'+datasetname+'.dat', 'wb') as f:
            pickle.dump(blocks_dc_list, f)
    print('block number:' + str(len(blocks_list)))  #输出block的总数量
    if only_generate_conflict_blocks:
        return
    K, M, L = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]
    weight_sum = np.sum(source_weight)
    block_index = -1
    LCset_list = []
    Cells_list = []
    value_dict_map = dict()
    generate_start_time = time.clock()
    for i in range(M):
        for j in range(L):
            value_dict_map[(i,j)] = generate_value_dict(data_mat, source_weight, i, j)
    for block in blocks_list:  #blocks_list中存储了每个block
        block_index += 1
        NewConstraints = set()
        ClaimsInBlock_set_all = set()
        for dc in blocks_dc_list[block_index]:
            probability_dict = dict()
            ClaimsInBlock_set = set()
            for cla in dc.clause_list:
                atts = cla.translate(Att_list)
                if atts == None:
                    continue
                if cla.single_entity:
                    for entity in block:
                        ClaimsInBlock_set.add((entity,atts[0]))
                        if len(atts) == 2:
                            value_dictB = value_dict_map[(entity,atts[1])]
                            ClaimsInBlock_set.add((entity,atts[1]))
                        else:
                            value_dictB = cla.constant
                        entities = (entity,)
                        poss = calc_probability(weight_sum, value_dict_map[(entity,atts[0])], value_dictB, cla.op)
                        if (entities,cla) not in probability_dict:
                            probability_dict[(entities,cla)] = poss
                        else:
                            if poss < probability_dict[(entities,cla)]:
                                probability_dict[(entities,cla)] = poss
                else:
                    for t1 in block:
                        for t2 in block:
                            if t1 == t2:
                                continue
                            ClaimsInBlock_set.add((t1,atts[0]))
                            ClaimsInBlock_set.add((t2,atts[1]))

                            entities = (t1,t2)
                            poss = calc_probability(weight_sum, value_dict_map[(t1,atts[0])], value_dict_map[(t2,atts[1])], cla.op)
                            if entities not in probability_dict:
                                probability_dict[(entities, cla)] = poss
                            else:
                                if poss < probability_dict[entities][1]:
                                    probability_dict[(entities, cla)] = poss
            probability_list = sorted(probability_dict.items(), key=lambda x: 0 if x[0][1].op == '!=' else 1)
            probability_list = sorted(probability_list, key=lambda x: x[1])
            list_index = 0
            ClaimsInBlock_set_copy = ClaimsInBlock_set.copy()
            ClaimsInBlock_set_all = ClaimsInBlock_set_all.union(ClaimsInBlock_set_copy)
            vis_entities = set()
            while True:
                if list_index >= len(probability_list):
                    break
                entities, cla = probability_list[list_index][0]
                if entities in vis_entities:
                    list_index += 1
                    continue
                vis_entities.add(entities)
                atts = cla.translate(Att_list)
                if atts == None:
                    list_index += 1
                    continue
                cla_Involved_claims = cla.get_Involved_claims(entities,Att_list)
                valid = True
                NewConstraints.add(generate_constraints(entities, cla, Att_list))
                for claim in cla_Involved_claims:
                    if claim in ClaimsInBlock_set:
                        ClaimsInBlock_set.remove(claim)

                for dc_cla in dc.clause_list:
                    if dc_cla == cla:
                        continue
                    claims = dc_cla.get_Involved_claims(entities,Att_list)
                    if len(claims) == 2:
                        if (claims[0] not in cla_Involved_claims) and (claims[0] in ClaimsInBlock_set):
                            if (claims[1] not in cla_Involved_claims) and (claims[1] in ClaimsInBlock_set):
                                NewConstraints.add((claims[0][0],claims[0][1],claims[1][0],claims[1][1],dc_cla.op))
                                ClaimsInBlock_set.remove(claims[0])
                                ClaimsInBlock_set.remove(claims[1])
                    else:
                        if (claims[0] not in cla_Involved_claims) and (claims[0] in ClaimsInBlock_set):
                            NewConstraints.add((claims[0][0],claims[0][1],dc_cla.constant,dc_cla.op))
                            ClaimsInBlock_set.remove(claims[0])

                list_index += 1
        Cells_list.append(ClaimsInBlock_set_all)
        LCset_list.append(NewConstraints)
    generate_end_time = time.clock()
    generate_time = generate_end_time - generate_start_time
    return Cells_list, LCset_list, Partition_time, generate_time

















