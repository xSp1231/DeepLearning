import numpy as np
import matplotlib.pyplot as plt
import torch as t
from torch.autograd import Variable as var

def get_data(x,w,b,d):
    c,r = x.shape
    y = (w * x * x + b*x + d)+ (0.1*(2*np.random.rand(c,r)-1))
    return(y)

xs = np.arange(0,3,0.01).reshape(-1,1)
ys = get_data(xs,1,-2,3)

xs = var(t.Tensor(xs))
ys = var(t.Tensor(ys))