import torch.nn as nn
import torch.nn.functional as Func


class A_net(nn.Module):
    def __init__(self, state_n, action_n, net_width):
        super(A_net, self).__init__()
        self.A = nn.Sequential(
            nn.Linear(state_n, net_width),
            nn.Tanh(),
            nn.Linear(net_width, net_width),
            nn.Tanh(),
        )
        self.act = nn.Sequential(
            nn.Linear(net_width, action_n),
            nn.Identity()
        )
        self.Aux_v = nn.Sequential(
            nn.Linear(net_width, 1),
            nn.Identity()
        )

    def forward(self, s):
        n = self.A(s)
        return n

    def prob(self, state, S_dim=-1):
        n = self.forward(state)
        prob = Func.softmax(self.act(n), S_dim)
        return prob

    def get_aux_v(self, state):
        n = self.forward(state)
        aux_v = self.Aux_v(n)
        return aux_v


class C_net(nn.Module):
    def __init__(self, state_n, net_width):
        super(C_net, self).__init__()
        self.C = nn.Sequential(
            nn.Linear(state_n, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 1),
            nn.Identity()
        )

    def forward(self, state):
        V = self.C(state)
        return V
