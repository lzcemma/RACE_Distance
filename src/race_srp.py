import numpy as np
from numba import guvectorize
from numba import float64, intp
import torch

def get_race_distance(a, b):
    return np.linalg.norm(a-b, axis=1).mean()


class SRP_Gaussin_torch():
    def __init__(self, K, R, d, seed):
        self.N = K * R  # number of hashes
        self.d = d  # data dimension
        self.K = K
        self.R = R
        np.random.seed(seed)
        self.W = torch.from_numpy(np.random.normal(size = (self.N, d))).float()
        self.powersOfTwo = torch.from_numpy(np.array([2**i for i in range(self.K)])).float()

    def hash(self,x, device):
        output = torch.sign(torch.matmul(x.to(device), self.W.to(device).T))
        output = torch.gt(output, 0).float()
        output = output.reshape(-1,self.R, self.K)
        return torch.matmul(output, self.powersOfTwo.to(device)).int().cpu().numpy()
        
class RACE_SRP():
    @guvectorize([( intp[:,:], float64[:], float64[:,:], float64[:,:])], '(n,l1),(l2),(m,k)->(m,k)',
                 target = "parallel", nopython=True,cache = True)
    def increasecount(hashcodes, alpha, zeros, out):
        out[:,:] = 0.0
        for i in range(out.shape[0]):
            for j in range(hashcodes.shape[0]):
                out[i, hashcodes[j,i]] += alpha[j]

    def __init__(self, repetitions, num_hashes, dimension, hashes, dtype = np.float32):
        self.dtype = dtype
        self.R = repetitions # number of ACEs (rows) in the array
        self.W = 2 ** num_hashes  # range of each ACE (width of each row)
        self.K = num_hashes
        self.D = dimension
        self.N = 0
        self.counts = np.zeros((self.R,self.W),dtype = self.dtype)
        self.hashes = hashes
    
    # increase count(weight) for X (batchsize * dimension)
    def add(self, X, alpha, device):
        self.N += X.shape[0]
        hashcode = self.hashes.hash(X, device)
        if(device == "cuda"):
            hashcode = hashcode.cpu().numpy()
        self.counts = self.increasecount(hashcode, alpha, self.counts)
            
    def print_table(self):
        print(self.counts.round(decimals=2))
    

#testing
# K = 3
# D = 4
# R = 2
# race = RACE_SRP(R, K, D)
# srp_hash = SRP(K, D, 1)
# X = rng.randn(3, D)
# x_test = rng.randn(5, D)
# alpha = np.array([0.4,0.2,0.3])
# race.add(X, alpha)
# for x in x_test:
#     race.query(x)
#     print()