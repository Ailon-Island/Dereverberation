import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

Q4 = torch.tensor(np.array(pd.read_excel("./Q4.xlsx", header=None))).reshape(1, 1,-1).to('cuda')
print(Q4.shape)
lr = 1E-1
T = 844

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # self.bn = nn.BatchNorm2d(1)
        self.alpha0 = torch.nn.Parameter(torch.tensor(0.4886).to('cuda'), requires_grad=True)
        self.alpha1 = torch.nn.Parameter(torch.tensor(0.3893).to('cuda'), requires_grad=True)
        torch.autograd.set_detect_anomaly(True)

    def forward(self, q):
        T0 = 844
        T1 = 1264
        length = 8137

        A = torch.diag(torch.ones((length),device='cuda', dtype=torch.double))
        A = Variable(A, requires_grad=True)
        A = A + torch.diag(torch.ones((length-T0),device='cuda') * self.alpha0, T0)
        A = A + torch.diag(torch.ones((length-T1),device='cuda') * self.alpha1, T1)
        A = torch.inverse(A)

        x = q @ A

        loss = torch.sum(x[0,0,-T1:] ** 2)
        return loss

model = NeuralNetwork().to('cuda')

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
lr_scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

min_loss = np.inf
breakcount = 0
for i in range(100000):
    optimizer.zero_grad()
    loss = model(Q4)
    if loss.item() < min_loss :
        min_loss = loss.item()
        breakcount = 0
    else:
        breakcount = breakcount + 1
    loss.backward()
    print(model.alpha0.grad, model.alpha1.grad)
    optimizer.step()
    print(model.alpha0.item(), model.alpha1.item())
    print("loss: {}".format(loss))
    print("breakcount: {}".format(breakcount))
    if breakcount > 100:
        break
