class ClusterPair:
    def __init__(self,a,b):
        self.a = a.copy()
        self.b = b.copy()

    @classmethod
    def getNewClusterPair(cls,n = 0):
        return cls(set(i for i in range(n)),set(i for i in range(n)))

    def __str__(self):
        return '(' + str(self.a) + ',' + str(self.b) + ')'

    # def union(self, o):
    #     return ClusterPair(self.a.union(o.a),self.b.union(o.b))
    #
    # def intersection(self, o):
    #     return ClusterPair(self.a.intersection(o.a),self.b.intersection(o.b))

    def isEmpty(self):
        # print('??',self.a,self.b)
        # print('??',len(self.a))
        # print('??',type(self.a))
        if len(self.a) == 0 or len(self.b) == 0:
            return True
        if len(self.a) == 1 and len(self.b) == 1 and len(self.a-self.b) == 0:
            return True
        return False

# class ClusterPairsList:
#     def __init__(self, list):
#         self.cpList = list.copy()
#
#     @classmethod
#     def getInitial(cls,n = 0):
#         assert n >= 0
#         if n == 0:
#             return cls([])
#         a = ClusterPair.getNewClusterPair(n)
#         return cls([a])
#
#     def __str__(self):
#         s = str('[')
#         for i in range(len(self.cpList)):
#             if i > 0:
#                 s += '; '
#             s += str(self.cpList[i])
#         return s + ']'
#
#     def intersection(self, o):
#         cpList = list()
#         for cp1 in self.cpList:
#             for cp2 in o.cpList:
#                 # print(cp1,cp2)
#                 a = cp1.intersection(cp2)
#                 if not a.isEmpty():
#                     cpList.append(a)
#                     # print('a',str(a))
#         return ClusterPairsList(cpList)
#
#     def add(self,cp):
#         assert type(cp) != 'ClusterPair'
#         self.cpList.append(cp)
#
#     def isEmpty(self):
#         return len(self.cpList) == 0

# if __name__ == '__main__':
#     a = ClusterPairsList.getInitial()
#     a.add(ClusterPair({1, 2, 3}, {3, 4, 5}))
#     b = ClusterPairsList.getInitial()
#     b.add(ClusterPair.getNewClusterPair(4))
#
#     print(str(a))
#     # print('--')
#     print(str(b))
#     print(a.intersection(b))


