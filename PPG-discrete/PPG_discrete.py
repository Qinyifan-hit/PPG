import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical
import copy
import math
from Net_con import A_net, C_net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPG_Agent(object):
    def __init__(self, opt):
        self.action_n = opt.action_dim
        self.state_n = opt.state_dim
        self.Actor = A_net(self.state_n, self.action_n, opt.net_width).to(device)
        self.A_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=opt.lr)
        self.Critic = C_net(self.state_n, opt.net_width).to(device)
        self.C_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=opt.lr)

        self.gamma = opt.gamma
        self.lambd = opt.lambd
        self.clip_rate = opt.clip_rate
        self.l2_reg = opt.l2_reg
        self.batch_size = opt.batch_size
        self.K_epochs = opt.K_epochs
        self.N_pi = opt.N_pi
        self.T_horizon = opt.T_horizon
        self.epochs_aux = opt.epochs_aux

        self.B_s = torch.zeros((self.N_pi, self.T_horizon, self.state_n), dtype=torch.float32).to(device)
        self.B_V = torch.zeros((self.N_pi, self.T_horizon, 1), dtype=torch.float32).to(device)
        self.B_ind = 0

    def action_select(self, s, determine):
        state = torch.FloatTensor(s).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = self.Actor.prob(state, S_dim=-1)
            if determine:
                action = torch.argmax(prob).item()
            else:
                Cate = Categorical(prob)
                action = Cate.sample().item()

            prob_a = None if determine else prob[0, action].item()
            return action, prob_a

    def train(self, traj):
        s, a, r, s_, prob_a, done = traj.read()
        # TD(lambd) --> Adv // Q
        with torch.no_grad():
            traj_len = len(done)
            t_turns = math.ceil(traj_len / self.batch_size)
            V = self.Critic(s)
            V_ = self.Critic(s_)
            delta = (r + ~done * self.gamma * V_ - V)
            delta = delta.cpu().view(1, -1).squeeze(0).numpy()
            adv = [0]

            done = done.cpu().view(1, -1).squeeze(0).numpy()
            for j in range(traj_len - 1, -1, -1):
                adv_j = delta[j] + self.gamma * self.lambd * adv[-1] * (~done[j])
                adv.append(adv_j)
            adv = adv[::-1]
            adv = copy.deepcopy(adv[0:traj_len])
            adv = torch.FloatTensor(adv).unsqueeze(-1).to(device)
            V_target = adv + V

            # A-C update: random order, small batch training
        for t in range(self.K_epochs):
            Ind = np.arange(traj_len)
            np.random.shuffle(Ind)
            Ind = torch.LongTensor(Ind).to(device)
            s, a, r, s_, prob_a, adv, V_target = s[Ind].clone(), a[Ind].clone(), r[Ind].clone(), s_[Ind].clone(), \
                prob_a[Ind].clone(), adv[Ind].clone(), V_target[Ind].clone()

            for j in range(t_turns):
                Ind_batch = slice(j * self.batch_size, min((j + 1) * self.batch_size, traj_len))
                prob = self.Actor.prob(s[Ind_batch], S_dim=1)
                prob_a_c = torch.gather(prob, 1, a[Ind_batch].long())

                r_t = torch.exp(torch.log(prob_a_c) - torch.log(prob_a[Ind_batch]))
                L1 = r_t * adv[Ind_batch]
                L2 = torch.clip(r_t, 1 - self.clip_rate, 1 + self.clip_rate) * adv[Ind_batch]
                A_loss = -torch.mean(torch.min(L1, L2))

                self.A_optimizer.zero_grad()
                A_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), 40)
                self.A_optimizer.step()

                C_loss = F.mse_loss(self.Critic(s[Ind_batch]), V_target[Ind_batch])
                for name, param in self.Critic.named_parameters():
                    if 'weight' in name:
                        C_loss += param.pow(2).sum() * self.l2_reg
                self.C_optimizer.zero_grad()
                C_loss.backward()
                self.C_optimizer.step()

        self.B_s[self.B_ind, :] = s
        self.B_V[self.B_ind, :] = V_target
        self.B_ind = (self.B_ind + 1) % self.N_pi

    def aux_train(self):
        mem_len = int(self.T_horizon * self.N_pi)
        Aux_s = self.B_s.reshape((mem_len, self.state_n)).to(device).detach()
        Aux_v = self.B_V.reshape((mem_len, 1)).to(device).detach()

        probs_old = self.Actor.prob(Aux_s).detach()
        Ind_aux = np.arange(mem_len)
        e_turns = int(mem_len / self.batch_size)
        for _ in range(self.epochs_aux):
            np.random.shuffle(Ind_aux)
            probs_old, Aux_s, Aux_v = probs_old[Ind_aux].clone(), Aux_s[Ind_aux].clone(), Aux_v[Ind_aux].clone()
            for j in range(e_turns):
                Ind_aux_b = slice(j * self.batch_size, min(mem_len, (j + 1) * self.batch_size))
                V_aux = self.Actor.get_aux_v(Aux_s[Ind_aux_b])
                probs_new = self.Actor.prob(Aux_s[Ind_aux_b])
                new_pi = Categorical(probs=probs_new)
                old_pi = Categorical(probs=probs_old[Ind_aux_b])

                Aux_loss = 0.5 * F.mse_loss(V_aux, Aux_v[Ind_aux_b])
                Kl_loss = torch.distributions.kl_divergence(old_pi, new_pi).mean()
                Joint_loss = Aux_loss + Kl_loss
                self.A_optimizer.zero_grad()
                Joint_loss.backward()
                self.A_optimizer.step()

                V = self.Critic(Aux_s[Ind_aux_b])
                V_loss = 0.5 * F.mse_loss(V, Aux_v[Ind_aux_b])
                self.C_optimizer.zero_grad()
                V_loss.backward()
                self.C_optimizer.step()

    def save(self, EnvName, timestep):
        torch.save(self.Actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName, timestep))
        torch.save(self.Critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName, timestep))

    def load(self, EnvName, timestep):
        self.Actor.load_state_dict(
            torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=device))
        self.Critic.load_state_dict(
            torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep), map_location=device))


class traj_memory(object):
    def __init__(self, T_horizon, state_n):
        self.T_horizon = T_horizon
        self.s = np.zeros((self.T_horizon, state_n), dtype=np.float32)
        self.a = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.r = np.zeros((self.T_horizon, 1), dtype=np.float32)
        self.s_ = np.zeros((self.T_horizon, state_n), dtype=np.float32)
        self.done = np.zeros((self.T_horizon, 1), dtype=np.bool_)

        self.prob_a = np.zeros((self.T_horizon, 1), dtype=np.float32)

    def add(self, s, a, prob_b, r, s_, done, traj_len):
        self.s[traj_len] = s
        self.a[traj_len] = a
        self.prob_a[traj_len] = prob_b
        self.r[traj_len] = r
        self.s_[traj_len] = s_

        self.done[traj_len] = done

    def read(self):
        return (
            torch.FloatTensor(self.s).to(device),
            torch.Tensor(self.a).to(device),
            torch.FloatTensor(self.r).to(device),
            torch.FloatTensor(self.s_).to(device),
            torch.FloatTensor(self.prob_a).to(device),
            torch.BoolTensor(self.done).to(device),
        )
