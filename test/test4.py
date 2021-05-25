"""


Author: Tong
Time: --2021
"""
import torch
from torch.nn.modules.rnn import LSTM
import numpy as np

a = torch.ones([2, 20], ) * 2

b = torch.ones([1, 20]) * 3

c = torch.cat([a, b], dim=0)

print(torch.unique(c))

d = torch.tensor([1, 2], dtype=torch.int64)

print(c[d])

c1 = torch.tensor([2, 3, 3], dtype=torch.int64)

e = torch.where(c1 == 3, True, False)

f = c[e]

print(f)

g = torch.mean(f, dim=0)

print(g)

a = torch.ones(10)
b = torch.zeros(10)
c = torch.stack([a,b], dim=0)
print(c.shape)
print(c)

c[0][2] = 10

a, b = torch.max(c, dim=1)
print(a)
print(b)

print(torch.tensor(False))

x = torch.randn(10)
print(x)
_, pred = torch.max(x, 0)

print(pred)


a = np.arange(10)
print(a)
b = np.max(a)
print(b)

a = np.arange(12*15)
print(a)
a = np.reshape(a, (15, 12))
print(a)
print(np.mean(a, axis=1))
print(a.std(axis=1))

a = torch.tensor(10)
if a == 10:
    print(True)
    



