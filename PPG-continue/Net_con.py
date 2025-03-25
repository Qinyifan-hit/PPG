import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Normal


class A_net_beta(nn.Module):
    def __init__(self, action_n, state_n, net_width):
        super(A_net_beta, self).__init__()
        self.A = nn.Sequential(
            nn.Linear(state_n, net_width),
            nn.Tanh(),
            nn.Linear(net_width, net_width),
            nn.Tanh()
        )
        self.alpha = nn.Sequential(*[nn.Linear(net_width, action_n), nn.Identity()])
        self.beta = nn.Sequential(*[nn.Linear(net_width, action_n), nn.Identity()])
        self.V_aux = nn.Sequential(*[nn.Linear(net_width, 1), nn.Identity()])

    def froward(self, s):
        p = self.A(s)
        alpha = F.softplus(self.alpha(p)) + 1.0
        beta = F.softplus(self.beta(p)) + 1.0
        return alpha, beta

    def get_distri(self, s):
        alpha, beta = self.froward(s)
        distri = Beta(alpha, beta)  # Beta Distribution (0-1) for each dim of action
        return distri

    def get_action(self, s):
        alpha, beta = self.froward(s)
        a = alpha / (alpha + beta)  # 0-1
        return a

    def get_aux_value(self, s):
        p = self.A(s)
        V = self.V_aux(p)
        return V


class C_net(nn.Module):
    def __init__(self, state_n, net_width):
        super(C_net, self).__init__()
        self.V = nn.Sequential(
            nn.Linear(state_n, net_width),
            nn.ReLU(),
            nn.Linear(net_width, net_width),
            nn.ReLU(),
            nn.Linear(net_width, 1),
            nn.Identity()
        )

    def forward(self, s):
        V = self.V(s)
        return V
