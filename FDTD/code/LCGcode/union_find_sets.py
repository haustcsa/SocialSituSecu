class union_find_sets:
    def __init__(self, N):
        self.N = N
        self.pre = [i for i in range(N)]
    
    def find(self, x):
        r = x
        while self.pre[r] != r:
            r = self.pre[r]
        i = x
        j = -1
        while i != r:
            j = self.pre[i]
            self.pre[i] = r
            i = j
        return r
    
    def union(self, x, y):
        a = self.find(x)
        b = self.find(y)
        if a != b:
            self.pre[a] = b

    
    def generate_sets(self):
        sets_dict = dict()
        for i in range(self.N):
            pre_i = self.find(i)
            if pre_i not in sets_dict:
                sets_dict[pre_i] = set()
            sets_dict[pre_i].add(i)
        return sets_dict.values()