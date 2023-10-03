import re
from Find_FD import findFD

Invalid_Sign = -1


class clause:

    def __init__(self):
        self.op = ''
        self.left = ''
        self.right = ''
        self.single_entity = False
        self.with_constant = False
        self.constant = 0

    def __compare__(self, op, valueA, valueB):
        if (valueA == Invalid_Sign) or (valueB == Invalid_Sign):
            return True
        if op == '!=':
            return valueA != valueB
        if op == '=':
            return valueA == valueB
        if op == '<':
            return valueA < valueB
        if op == '>':
            return valueA > valueB
        return True

    def __str__(self):
        if len(self.op) == 0:
            return ''
        if self.single_entity:
            if self.with_constant:
                return 't1.' + self.left + self.op + str(self.constant)
            else:
                return 't1.' + self.left + self.op + 't1.' + self.right
        else:
            return 't1.' + self.left + self.op + 't2.' + self.right

    def __hash__(self):
        return hash(self.__str__())

    def __eq__(self, other):
        if isinstance(other, clause):
            return self.__str__() == str(other)
        else:
            return False

    def __lt__(self, other):
        return self.__str__() < str(other)

    def __gt__(self, other):
        return self.__str__() > str(other)

    def getConstant(self):
        if not self.with_constant:
            return None
        return self.constant

    def translate(self, Att_list):
        if self.single_entity:
            if self.left not in Att_list:
                return None
            att1 = Att_list.index(self.left)
            if self.with_constant:
                return (att1,)
            else:
                if self.right not in Att_list:
                    return None
                att2 = Att_list.index(self.right)
                return (att1, att2)
        else:
            if self.left not in Att_list:
                return None
            if self.right not in Att_list:
                return None
            att1 = Att_list.index(self.left)
            att2 = Att_list.index(self.right)
            return (att1, att2)

    def get_Involved_claims(self, entities, Att_list):
        atts = self.translate(Att_list)
        if atts == None:
            return None
        Involved_claims = []
        if self.single_entity:
            Involved_claims.append((entities[0], atts[0]))
            if len(atts) == 2:
                Involved_claims.append((entities[0], atts[1]))
        else:
            Involved_claims.append((entities[0], atts[0]))
            if len(entities) > 1:
                Involved_claims.append((entities[1], atts[1]))
        return Involved_claims

    def check(self, Att_list, claimA, claimB=None):
        if self.single_entity:
            if self.left not in Att_list:
                return True
            att1 = Att_list.index(self.left)
            valueA = claimA[att1]
            if self.with_constant:
                return self.__compare__(self.op, valueA, self.constant)
            else:
                if self.right not in Att_list:
                    return True
                att2 = Att_list.index(self.right)
                valueB = claimA[att2]
                return self.__compare__(self.op, valueA, valueB)
        else:
            if self.left not in Att_list:
                return True
            if self.right not in Att_list:
                return True
            att1 = Att_list.index(self.left)
            valueA = claimA[att1]
            att2 = Att_list.index(self.right)
            valueB = claimB[att2]
            return self.__compare__(self.op, valueA, valueB)

    def MayBeFalse(self, t1, t2, data_mat, Att_list):
        if self.single_entity:
            if self.left not in Att_list:
                return False
            att1 = Att_list.index(self.left)
            if self.with_constant:
                value_list = data_mat[:, t1, att1]
                value_list = value_list[value_list != Invalid_Sign]
                for value in value_list:
                    if not self.__compare__(self.op, value, self.constant):
                        return True
                return False
            else:
                if self.right not in Att_list:
                    return False
                att2 = Att_list.index(self.right)
                value_list1 = data_mat[:, t1, att1]
                value_list1 = value_list1[value_list1 != Invalid_Sign]
                value_list2 = data_mat[:, t1, att2]
                value_list2 = value_list2[value_list2 != Invalid_Sign]
                for valueA in value_list1:
                    for valueB in value_list2:
                        if not self.__compare__(self.op, valueA, valueB):
                            return True
                return False
        else:
            if self.left not in Att_list:
                return False
            if self.right not in Att_list:
                return False
            att1 = Att_list.index(self.left)
            att2 = Att_list.index(self.right)
            value_list1 = data_mat[:, t1, att1]
            value_list1 = value_list1[value_list1 != Invalid_Sign]
            value_list2 = data_mat[:, t2, att2]
            value_list2 = value_list2[value_list2 != Invalid_Sign]
            for valueA in value_list1:
                for valueB in value_list2:
                    if not self.__compare__(self.op, valueA, valueB):
                        return True
            return False

    def MayBeTrue(self, t1, t2, data_mat, Att_list):
        if self.single_entity:
            if self.left not in Att_list:
                return False
            att1 = Att_list.index(self.left)
            if self.with_constant:
                value_list = data_mat[:, t1, att1]
                value_list = value_list[value_list != Invalid_Sign]
                ans = False
                for value in value_list:
                    if self.__compare__(self.op, value, self.constant):
                        ans = True
                        break
                return ans
            else:
                if self.right not in Att_list:
                    return False
                att2 = Att_list.index(self.right)
                value_list1 = data_mat[:, t1, att1]
                value_list1 = value_list1[value_list1 != Invalid_Sign]
                value_list2 = data_mat[:, t1, att2]
                value_list2 = value_list2[value_list2 != Invalid_Sign]
                ans = False
                for valueA in value_list1:
                    for valueB in value_list2:
                        if self.__compare__(self.op, valueA, valueB):
                            ans = True
                            break
                return ans
        else:
            if self.left not in Att_list:
                return False
            if self.right not in Att_list:
                return False
            att1 = Att_list.index(self.left)
            att2 = Att_list.index(self.right)
            value_list1 = data_mat[:, t1, att1]
            value_list1 = value_list1[value_list1 != Invalid_Sign]
            value_list2 = data_mat[:, t2, att2]
            value_list2 = value_list2[value_list2 != Invalid_Sign]
            ans = False
            for valueA in value_list1:
                for valueB in value_list2:
                    if self.__compare__(self.op, valueA, valueB):
                        ans = True
                        break
            return ans


def clause_parser(new_clause_str, encoding_dict_list, Att_list, IsContinuous):
    new_clause_str = new_clause_str.strip()
    op = ''
    op_list = ['!=', '<=', '>=', '>', '<', '=']
    if '!=' in new_clause_str:
        op = '!='
    else:
        for o in op_list:
            if o in new_clause_str:
                op = o
                break
    if len(op) == 0:
        return None
    new_clause = clause()
    new_clause.op = op
    left, right = new_clause_str.split(op)[0].strip(), new_clause_str.split(op)[1].strip()
    left_entity, left_att = left.split('.')[0], left.split('.')[1]
    new_clause.left = left_att
    if 't1' not in right and 't2' not in right:
        new_clause.single_entity = True
        new_clause.with_constant = True
        # new_clause.right = right
        att_num = Att_list.index(left_att)
        # if IsContinuous[att_num]:
        #     new_clause.constant = float(right)
        # else:
        if isinstance(encoding_dict_list, dict):
            encoding_dict = encoding_dict_list
        else:
            encoding_dict = encoding_dict_list[att_num]
        if right not in encoding_dict:
            # return None
            encoding_dict[right] = len(encoding_dict)
        new_clause.constant = encoding_dict[right]  # encode
    else:
        right_entity, right_att = right.split('.')[0], right.split('.')[1]
        new_clause.with_constant = False
        if right_entity == left_entity:
            new_clause.single_entity = True
        else:
            new_clause.single_entity = False
        new_clause.right = right_att
    return new_clause


class DC:

    def __init__(self):
        self.clause_list = []
        self.coverage = 0.0
        self.single_entity = -1

    def __str__(self):
        if len(self.clause_list) == 0:
            return ''
        ans = str(self.coverage) + ':not(' + str(self.clause_list[0])
        for i in range(1, len(self.clause_list)):
            ans += ('&' + str(self.clause_list[i]))
        ans += ')'
        return ans

    def __iter__(self):
        return self

    def __next__(self):
        return self.clause_list.__next__()

    def __len__(self):
        return len(self.clause_list)

    def add_clauses(self, cla):
        if cla not in self.clause_list:
            self.clause_list.append(cla)

    def sort_clauses(self, cmp):
        # self.clause_list = sorted(self.clause_list, key=lambda x : str(x))
        self.clause_list = sorted(self.clause_list, key=lambda x: cmp.val[x])

    def get_Involved_claims(self, entities, Att_list):
        Involved_claims_set = set()
        for cla in self.clause_list:
            Involved_claims = cla.get_Involved_claims(entities, Att_list)
            for claim in Involved_claims:
                Involved_claims_set.add(claim)
        return Involved_claims_set

    def Is_Single_Entity(self):
        if self.single_entity > -1:
            return self.single_entity == 1
        Is_single_entity = True
        for cla in self.clause_list:
            if not cla.single_entity:
                Is_single_entity = False
                break
        if Is_single_entity:
            self.single_entity = 1
        else:
            self.single_entity = 0
        return Is_single_entity

    def Is_Conflicting(self, Att_list, claimA, claimB):
        for cla in self.clause_list:
            if not cla.check(Att_list, claimA, claimB):
                return False
        for cla in self.clause_list:
            if not cla.check(Att_list, claimB, claimA):
                return False
        return True

    def Is_Potentially_Conflicting(self, t1, t2, data_mat, Att_list):  # all may be true
        ans = True
        for cla in self.clause_list:
            if not cla.MayBeTrue(t1, t2, data_mat, Att_list):
                ans = False
                break
        if ans:
            return ans
        ans = True
        for cla in self.clause_list:
            if not cla.MayBeTrue(t2, t1, data_mat, Att_list):
                ans = False
                break
        return ans


def DCs_str_parser(dc_str, encoding_dict_list, Att_list, IsContinuous):
    pattern = re.compile(r'([0-9.]+)[ ]*:[ ]*not[ ]*\([ ]*([^&]+)(([ ]*&[^&]+)*)\)[ ]*')
    m = pattern.match(dc_str)
    if m is None:
        return None
    new_DC = DC()
    new_DC.coverage = float(m.group(1))
    new_cla = clause_parser(m.group(2), encoding_dict_list, Att_list, IsContinuous)
    if new_cla is None:
        return None
    new_DC.clause_list.append(new_cla)
    if len(m.group(3)) > 0:
        clauses = m.group(3).strip('&').strip().strip('&').strip()
        not_valid = False
        for c in clauses.split('&'):
            new_cla = clause_parser(c, encoding_dict_list, Att_list, IsContinuous)
            if new_cla is None:
                not_valid = True
                return None
            new_DC.clause_list.append(new_cla)
        if not_valid:
            return None
    return new_DC


# def DCs_parser(filename, encoding_dict_list, Att_list, IsContinuous):
#     DC_list = []
#     pattern = re.compile(r'([0-9.]+)[ ]*:[ ]*not[ ]*\([ ]*([^&]+)(([ ]*&[^&]+)*)\)[ ]*')
#     with open(filename, 'r') as f:
#         for line in f:
#             m = pattern.match(line)
#             if m is None:
#                 continue
#             new_DC = DC()
#             new_DC.coverage = float(m.group(1))  # 列出第一个括号匹配部分
#             new_cla = clause_parser(m.group(2), encoding_dict_list, Att_list, IsContinuous)
#             if new_cla is None:
#                 continue
#             new_DC.clause_list.append(new_cla)
#             if len(m.group(3)) > 0:
#                 clauses = m.group(3).strip('&').strip().strip('&').strip()
#                 not_valid = False
#                 for c in clauses.split('&'):
#                     new_cla = clause_parser(c, encoding_dict_list, Att_list, IsContinuous)
#                     if new_cla is None:
#                         not_valid = True
#                         break
#                     new_DC.clause_list.append(new_cla)
#                 if not_valid:
#                     continue
#             DC_list.append(new_DC)
#     return DC_list

def DCs_parser(encoding_dict_list, Att_list, IsContinuous):
    DC_list = []
    pattern = re.compile(r'([0-9.]+)[ ]*:[ ]*not[ ]*\([ ]*([^&]+)(([ ]*&[^&]+)*)\)[ ]*')
    dcs = findFD()
    for line in dcs:
        m = pattern.match(line)
        if m is None:
            continue
        new_DC = DC()
        new_DC.coverage = float(m.group(1))  # 列出第一个括号匹配部分
        new_cla = clause_parser(m.group(2), encoding_dict_list, Att_list, IsContinuous)
        if new_cla is None:
            continue
        new_DC.clause_list.append(new_cla)
        if len(m.group(3)) > 0:
            clauses = m.group(3).strip('&').strip().strip('&').strip()
            not_valid = False
            for c in clauses.split('&'):
                new_cla = clause_parser(c, encoding_dict_list, Att_list, IsContinuous)
                if new_cla is None:
                    not_valid = True
                    break
                new_DC.clause_list.append(new_cla)
            if not_valid:
                continue
        DC_list.append(new_DC)
    return DC_list
