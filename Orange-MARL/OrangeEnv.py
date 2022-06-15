import numpy as np
from numpy.lib.shape_base import split
import OrangeNet 


class MultiAgentEnv():
    
    def __init__(self,n, world = 'lattice'):
        
        if (world == "lattice"):
            self.world = OrangeNet.latticeNet(n)
        elif (world == "random"):
            self.world = OrangeNet.randomNet(n)
        elif (world == 'ba'):
            self.world = OrangeNet.BANet(n)
        elif (world == "ws"):
            self.world = OrangeNet.WSNet(n)
        else:
            print("Wrong Name!!!!!")    
        self.n = n
        self.action_space = [0,1]
        self.observation_space = []
        for i in range(self.n):
            self.observation_space.append(len(self.world.Nodes[i].neighbor))
        self.gameMatrix = np.array([[2,-1],[4,0]])
    
    def step(self, action_n):
        state = []
        # action_n = np.array(action_n)[:,0]
        obs_n = []
        reward_n = []
        done_n = [False] * self.n 
        info_n = {"n":[]}
        
        for i in range(self.n):
            #计算每个节点的收益
            payoffTemp = 0
            
            for j in self.world.Nodes[i].neighbor:
                payoffTemp += self.gameMatrix[int(action_n[i]), int(action_n[j])]
            reward_n.append(payoffTemp)
        reward_n = np.array(reward_n)
        
            
        for i in range(self.n):
            obs_ = np.array([
                            action_n[self.world.Nodes[i].neighbor]
                             ,reward_n[self.world.Nodes[i].neighbor]
                             ]
                            )
            obs_n.append(obs_.flatten())
        
        state = np.array([action_n, reward_n],
                         #dtype=object
                         ).flatten()
       
        
        # for simplicity, the .flatten() function is used to make the state and the obs_n into one dimension.
        
        #If you want to use a two-dimensional matrix, use the .reshape(2,-1).
        
        return state, obs_n, reward_n, done_n, info_n
    
    
    
    
    def reset(self, init = 0.5): 
        state = []
        obs_n = []
        # action_n = np.random.randint(2,size = self.n)  
        #该命令用于生成完全随机的动作空间，先暂时不用
        
        action_n = np.zeros(self.n)
        split = int(init * self.n)
        action_n[split: ] = 1.0
        reward_n = []
        for i in range(self.n):
            #计算每个节点的收益
            payoffTemp = 0
            
            for j in self.world.Nodes[i].neighbor:
                payoffTemp += self.gameMatrix[int(action_n[i]), int(action_n[j])]
            reward_n.append(payoffTemp)
        reward_n = np.array(reward_n)
        
        for i in range(self.n):
            
            obs_ = np.array([
                            action_n[self.world.Nodes[i].neighbor]
                             ,reward_n[self.world.Nodes[i].neighbor]
                             ]
                            )
            obs_n.append(obs_.flatten())

        state =np.array([action_n, reward_n]).flatten()
        # state = torch.tensor(state,dtype=torch.float32).reshape(-1)
        # obs_n = torch.tensor(obs_n,dtype=torch.float32).reshape(-1)
        return state , obs_n
    