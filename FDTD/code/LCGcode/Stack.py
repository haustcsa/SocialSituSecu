import time

class Stack:
    def __init__(self):
        self.items = []
        
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if len(self.items) == 0:
            return None
        x = self.items[-1]
        self.items.pop()
        return x
    
    def isEmpty(self):
        return len(self.items) == 0
    
    def peek(self):
        return self.items[-1]
        
    def size(self):
        return len(self.items)

    def __len__(self):
        return len(self.items)