import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

Q3 = torch.tensor(np.array(pd.read_excel("./Q3.xlsx", header=None))).reshape(1, 1,-1).to('cuda')
print(Q3.shape)
lr = 1E-2
T = 755

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(0.7344,dtype=torch.double).to('cuda'), requires_grad=True)

    def forward(self, q):
        T = 755

        length = 6783
        size = np.int(np.ceil(length / T))
        A = torch.zeros((length , length), device='cuda', dtype=torch.double)
        A = Variable(A, requires_grad=True)

        for i in range(size):
            A = A + torch.diag(   torch.ones((length-i * T,),device='cuda', dtype=torch.double) * (-self.alpha)** i   , i * T,).to('cuda')
        x = q @ A
        # R = F.conv1d(x[:,:,:-T], (q-x)[:,:,T:]/self.alpha , padding=x[0,0,:-T].size()[0] - 1)
        # loss = -torch.max(R) / torch.sqrt(torch.sum(x[0,0,:-T]**2) * torch.sum((q-x)[0,0,T:]**2))
        loss = torch.sum((x.to('cpu')[0,0,-T:])**2).to('cuda')
        return loss

model = NeuralNetwork().to('cuda')

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

min_loss = np.inf
for i in range(100000):
    optimizer.zero_grad()
    loss = model(Q3)
    if loss.item() < min_loss :
        min_loss = loss.item()
        breakcount = 0
    else:
        breakcount = breakcount + 1
    loss.backward()
    print(model.alpha.grad)
    optimizer.step()
    print(model.alpha.item())
    print("loss: {}".format(loss))
    print("breakcount: {}".format(breakcount))
    if breakcount > 100:
        break
