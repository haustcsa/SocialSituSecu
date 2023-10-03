from PartitionRefiner import *
from ClusterPairs import *
from union_find_sets import union_find_sets


class TreeNode:

    def __init__(self, clause='', dcID = -1):
        self.clause = clause # clause。根节点是root
        self.son = dict() # key是clause，value是TreeNode
        self.dcID = dcID # -1表示不是终止节点，否则表示当前dc的编号
    
    def __contains__(self, item):
        return item in self.son

    def __hash__(self):
        return hash(str(self.clause))
    
    def __str__(self):
        return str(self.clause)

    def __eq__(self, other):
        return str(other) == str(self.clause)
    
    def __getitem__(self, item):
        return self.son[item]

    def __setitem__(self, key, value):
        self.son[key]=value

    def __delitem__(self, key):
        self.son.pop(key)

    def __delattr__(self, item):
        self.son.pop(item)

    def __len__(self):
        return len(self.son)

    def has_son(self, node):
        return node.clause in self.son
    
    def get_son(self, clause):
        return self.son[clause]

    def add_son(self, tree_node):
        if tree_node.clause in self.son:
            return
        self.son[tree_node.clause] = tree_node

class DCTree:
    def __init__(self, dc_set = None):
        self.rootNode = TreeNode(clause='root', dcID= -1)

    def add_DC(self, dc, id):
        tree_node = self.rootNode
        # print(dc)
        print("id: ", id,"   ",dc)
        for i in range(len(dc.clause_list)):
            print(dc.clause_list[i])
        for ind in range(len(dc.clause_list)):
            clause = dc.clause_list[ind]
            # if clause.single_entity:
            #     continue

            if clause in tree_node.son:
                tree_node = tree_node.son[clause]
            else:
                new_node = TreeNode(clause= clause, dcID= -1)
                if ind == len(dc.clause_list) - 1:
                    new_node.dcID = id
                    # print("addDC: ", ind)
                tree_node.add_son(new_node)
                tree_node = new_node


                # def add_DC(self, dc):
    #     dc.sort_clauses()
    #     tree_node = self.dc_tree
    #     for ind, clause in enumerate(dc):
    #         if clause in tree_node:
    #             tree_node = tree_node[clause]
    #         else:
    #             new_node = TreeNode(node_name = clause, is_dc = False)
    #             if ind == len(dc) - 1:
    #                 new_node.is_dc = True
    #             tree_node.add_son(new_node)



class DCTreeWorker:

    def __init__(self,tree, data_mat, Att_list):
        self.tree = tree
        self.data_mat = data_mat
        self.Att_list = Att_list
        self.edgeNumber = 0


    def work(self):
        tree = self.tree
        data_mat = self.data_mat
        Att_list = self.Att_list
        self.edgeNumber = 0

        K, L, M = data_mat.shape[0], data_mat.shape[1], data_mat.shape[2]
        # K个数据源，L个数据，M个属性

        self.union_set = union_find_sets(L) # 初始化并查集
        self.violateDcs = [set() for i in range(L)] # 每个点违反的dc集合
        # print("dfs start")
        # print(K,L,M)
        # print(data_mat)
        self.dfs(ClusterPair.getNewClusterPair(L), tree.rootNode)
        # print("dfs end")

        blocks_list = []  # 第一维是块数，第二维是该块的点的set
        blocks_dc_index_list = []  # 第二维是该块违反的dc的set

        rootSet = dict()
        now = 0
        for i in range(L):
            fa = self.union_set.find(i)
            if fa not in rootSet: # 如果找到一个新块
                rootSet[fa] = now # 记录当前块标号
                blocks_dc_index_list.append(set())
                blocks_list.append(set())
                now += 1
            id = rootSet[fa]
            blocks_list[id].add(i)
            for idIndex in self.violateDcs[i]: # 遍历i号点违反的dc的编号集合
                blocks_dc_index_list[id].add(idIndex)

        return blocks_list, blocks_dc_index_list

    def dfs(self, clusterPair, tree_node):
        if clusterPair.isEmpty():
            return
        # print("dfs : ", '(', len(clusterPair.a), len(clusterPair.b), ')',
        #       tree_node.clause, tree_node.dcID)

        if tree_node.dcID != -1: # 如果是一个dc
            id = tree_node.dcID
            for x in clusterPair.a:
                for y in clusterPair.b: # 枚举一对点
                    if x == y: continue
                    # print("merge: ",x+1,y+1)
                    self.union_set.union(x, y) # 合并
                    self.edgeNumber += 1
            z = clusterPair.a.union(clusterPair.b)
            for i in z:
                self.violateDcs[i].add(id)

        for clause in tree_node.son:
            node = tree_node.son[clause]
            cpList = PartitionRefiner.refine(clusterPair, clause, self.data_mat, self.Att_list)
            for cp in cpList:
                self.dfs(cp,node)
