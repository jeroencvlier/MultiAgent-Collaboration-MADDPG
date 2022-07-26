import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorPolicy(nn.Module):
    def __init__(self,state_size,action_size,hidden=[256,128],drop_out=0.25):
        super().__init__()
        self.hidden = hidden
        self.drop_out = drop_out
        layerlist = []
        layerlist.append(nn.Linear(state_size, hidden[0]))
        layerlist.append(nn.BatchNorm1d(hidden[0]))
        layerlist.append(nn.Dropout(p=drop_out))
        layerlist.append(nn.ReLU(inplace=True))
        layerlist.append(nn.Linear(hidden[0], hidden[1]))
        layerlist.append(nn.Dropout(p=drop_out))
        layerlist.append(nn.ReLU(inplace=True))
        layerlist.append(nn.Linear(hidden[1], action_size))
        layerlist.append(nn.Tanh())
        self.net = nn.Sequential(*layerlist)
        self.net.apply(init_weights)
        
    def forward(self, state):     
        x = np.array(state)
        x = self.net(torch.FloatTensor(x).to(device))
        return x
    
class CriticPolicy(nn.Module):
    def __init__(self,state_size,action_size,hidden=[256,128]):
        super().__init__()
        self.hidden = hidden
        input_size = state_size
        outout_size = 1
        layerlist = []
        layerlist.append(nn.Linear(state_size+action_size, hidden[0]))
        layerlist.append(nn.BatchNorm1d(hidden[0]))
        layerlist.append(nn.ReLU(inplace=True))
        layerlist.append(nn.Linear(hidden[0], hidden[1]))
        layerlist.append(nn.ReLU(inplace=True))
        layerlist.append(nn.Linear(hidden[1], outout_size))
        self.net = nn.Sequential(*layerlist)
        self.net.apply(init_weights)
    
    def forward(self, state, action):
        x = torch.FloatTensor(state)
        x = torch.cat((x,action), dim=1)
        x = self.net(x)
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
