import numpy as np

class nodes():
    
    def __init__(self,id_):
        self.id_ = id_  #节点的编号
        self.neighbor = []  # 记录节点的邻居  ps: 如果写成了self.neighbor = neighbor ，
        #上面为neighbor =[]就会出错，所以对象的neighbor 都会变成一个
        #a = nodes(1)
        # b = nodes(2)
        # print(id(a.neighbor) )
        # print(id(b.neighbor))
        # a.neighbor is b.neighbor
        #此为测试代码
        
class base_network:
    
    def __init__(self, vnum):
        self.vnum = vnum   #vnum: number of vertex  > 节点的数量 
        self.Nodes = self.initialize_null_nodes()
        self.edges = np.array([[0,1]])
    def initialize_null_nodes(self):
        Nodes = {} 
        for i in range(self.vnum):
            Nodes[i] = nodes(i)
        return Nodes

    def output(self):
        for i in range(self.vnum):
            print(self.Nodes[i].id_)
        print(self.Nodes)

class latticeNet(base_network):
    def __init__(self,vnum,limit = True):
        super().__init__(vnum)
        self.limit = limit
        self.create_network()
    def create_network(self):

        length = 0
        while(length**2 <= self.vnum):
            length += 1
        length -= 1
        for i in range(length):
            for j in range(length):
                no = i * length + j
                l = no - 1
                r = no + 1
                u = no - length
                d = no + length
                if(self.limit):
                    if(j != 0):
                        self.Nodes[no].neighbor.append(l)
                    if(j != length - 1):
                        self.Nodes[no].neighbor.append(r)    
                    if (i != 0):                    
                        self.Nodes[no].neighbor.append(u)
                    if (i != length - 1):
                        self.Nodes[no].neighbor.append(d)
                else:
                    if(j == 0):
                        l = l + length  
                    if(j == length - 1):
                        r = r - length      
                    if (i == 0):                    
                        u = u + length**2    
                    if (i == length - 1):
                        d = d - length**2
                    self.Nodes[no].neighbor.append(l)
                    self.Nodes[no].neighbor.append(r)
                    self.Nodes[no].neighbor.append(u)
                    self.Nodes[no].neighbor.append(d)
                    
                    
                    
class randomNet(base_network):
    def __init__(self,vnum,prob = 0.3):
        super().__init__(vnum)
        self.prob = prob
        self.random_pick()
    def random_pick(self):
        for i in range(self.vnum):
            for j in range(self.vnum):
                if (i != j) and ( j not in self.Nodes[i].neighbor )and( np.random.rand() < self.prob ):
                    self.Nodes[i].neighbor.append(j)
                    if (i not in self.Nodes[j].neighbor):
                        self.Nodes[j].neighbor.append(i)

 #for test                       
# r = random_network(10,0.2)
# for i in range (10):
#     print(str(i)+ ":" + str(r.Nodes[i].neighbor))

class WSNet(base_network):
    def __init__(self,vnum,prob = 0.6):
        super().__init__(vnum)
        self.K = 4
        self.prob = prob
        self.create_regular_ring()
        self.random_rewire()
    def create_regular_ring(self):
        for i in range(self.vnum):
            for j in range(int(self.K / 2)):
                left = i - j - 1
                right = i + j + 1
                if left < 0 :
                    left += self.vnum
                if right > self.vnum - 1:
                    right -= self.vnum
                self.Nodes[i].neighbor.append(left)
                self.Nodes[i].neighbor.append(right)

    def random_rewire(self):

        for i in range(self.vnum):
            if(np.random.rand() < self.prob):
                if_change = True
                while(if_change):
                    k = np.random.randint(self.vnum)
                    if (k not in self.Nodes[i].neighbor) and(k != i):
                        if_change = False 
                right = i + int(self.K / 2) 
                if right > self.vnum - 1:
                    right -= self.vnum
                self.Nodes[i].neighbor.remove(right)
                self.Nodes[right].neighbor.remove(i)
                self.Nodes[i].neighbor.append(k)
                self.Nodes[k].neighbor.append(i)

class BANet(base_network):
    def __init__(self,vnum):
        super().__init__(vnum)
        self.preferential_attachment()
    #@jit(nopython=True)
    def preferential_attachment(self):
        for i in range(self.vnum):
            if i > 1 :
                j = np.random.choice(np.arange(i-1))
                k = np.random.choice(self.edges[j,:])
                self.edges = np.row_stack((self.edges,[i,k]))
        for i in range(self.vnum):
            for item in self.edges:
                if item[0] == i:
                    self.Nodes[i].neighbor.append(item[1])
                if item[1] == i:
                    self.Nodes[i].neighbor.append(item[0])