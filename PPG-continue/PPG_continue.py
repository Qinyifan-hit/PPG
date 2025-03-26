import torch
import torch.nn.functional as F
import numpy as np
import torch.distributions
from Net_con import A_net_beta, C_net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPG_Agent(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.Actor = A_net_beta(self.action_dim, self.state_dim, self.net_width).to(device)
        self.A_optimizer = torch.optim.Adam(self.Actor.parameters(), lr=self.a_lr)
        self.Critic = C_net(self.state_dim, self.net_width).to(device)
        self.C_optimizer = torch.optim.Adam(self.Critic.parameters(), lr=self.c_lr)

        self.B_s = torch.zeros((self.N_pi, self.T_horizon, self.state_dim), dtype=torch.float32).to(device)
        self.B_V = torch.zeros((self.N_pi, self.T_horizon, 1), dtype=torch.float32).to(device)
        self.B_ind = 0

    def action_selection(self, s, iseval):
        s = torch.FloatTensor(s).to(device)
        with torch.no_grad():
            if iseval:
                a_p = self.Actor.get_action(s)
            else:
                Distri = self.Actor.get_distri(s)
                a_p = Distri.sample()
            prob_a = None if iseval else Distri.log_prob(a_p).cpu().numpy()
            return a_p.cpu().numpy(), prob_a

    def train(self, traj):
        # policy train
        self.entropy_coef *= self.entropy_coef_decay
        s, a, r, s_, prob_old, done, dw = traj.read()
        traj_len = s.shape[0]
        e_turns = int(traj_len / self.batch_size)
        with torch.no_grad():
            V = self.Critic(s)
            V_ = self.Critic(s_)
            delta = r + ~dw * self.gamma * V_ - V
            delta = delta.cpu().view(1, -1).squeeze(0).numpy()
            Adv = []

            done = done.cpu().view(1, -1).squeeze(0).numpy()
            A = 0.0
            for j in range(traj_len - 1, -1, -1):
                A = delta[j] + ~done[j] * self.gamma * self.lamda * A
                Adv.append(A)
            Adv.reverse()
            Adv = torch.FloatTensor(Adv).unsqueeze(-1).to(device)
            V_target = Adv + V
        Adv = (Adv - Adv.mean()) / (Adv.std() + 1e-5)

        for _ in range(self.epochs_p):
            Ind = np.arange(traj_len)
            np.random.shuffle(Ind)
            Adv, V_target, a, s, prob_old = Adv[Ind].clone(), V_target[Ind].clone(), a[Ind].clone(), s[Ind].clone(), \
                prob_old[Ind].clone()
            for j in range(e_turns):
                Ind_minb = np.arange(j * self.batch_size, min(traj_len, (j + 1) * self.batch_size))
                Distri_new = self.Actor.get_distri(s[Ind_minb])
                prob_new = Distri_new.log_prob(a[Ind_minb])
                prob_new_entropy = Distri_new.entropy().sum(dim=-1, keepdim=True)

                r_t = (prob_new.sum(dim=-1, keepdim=True) - prob_old[Ind_minb].sum(dim=-1, keepdim=True)).exp()
                L1 = r_t * Adv[Ind_minb]
                L2 = torch.clip(r_t, 1 - self.clip_rate, 1 + self.clip_rate) * Adv[Ind_minb]
                E_loss = -self.entropy_coef * prob_new_entropy.mean()
                L_clip = -(torch.min(L1, L2).mean())
                A_loss = E_loss + L_clip

                self.A_optimizer.zero_grad()
                A_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.Actor.parameters(), 40)
                self.A_optimizer.step()

                C_loss = 0.5 * F.mse_loss(self.Critic(s[Ind_minb]), V_target[Ind_minb])
                for name, param in self.Critic.named_parameters():
                    if 'weight' in name:
                        C_loss += param.pow(2).sum() * self.l2_reg
                self.C_optimizer.zero_grad()
                C_loss.backward()
                self.C_optimizer.step()

        # Store
        self.B_s[self.B_ind, :] = s
        self.B_V[self.B_ind, :] = V_target
        self.B_ind = (self.B_ind + 1) % self.N_pi

    def aux_train(self):
        mem_len = int(self.T_horizon * self.N_pi)
        Aux_s = self.B_s.reshape((mem_len, self.state_dim)).to(device).detach()
        Aux_V = self.B_V.reshape((mem_len, 1)).to(device).detach()

        Distri_old = self.Actor.get_distri(Aux_s)
        Distri_old_alpha = Distri_old.concentration1.detach()
        Distri_old_beta = Distri_old.concentration0.detach()

        Ind_aux = np.arange(mem_len)
        e_turns = int(mem_len / self.batch_size)
        for _ in range(self.epochs_aux):
            np.random.shuffle(Ind_aux)
            Aux_s, Aux_V, Distri_old_alpha, Distri_old_beta = Aux_s[Ind_aux].clone(), Aux_V[Ind_aux].clone(), \
                Distri_old_alpha[Ind_aux].clone(), Distri_old_beta[Ind_aux].clone()
            for j in range(e_turns):
                Ind_aux_b = slice(j * self.batch_size, min(mem_len, (j + 1) * self.batch_size))
                V_aux = self.Actor.get_aux_value(Aux_s[Ind_aux_b])
                Distri_new = self.Actor.get_distri(Aux_s[Ind_aux_b])
                'Aux_loss'
                Aux_loss = 0.5 * F.mse_loss(V_aux, Aux_V[Ind_aux_b])
                'Kl_loss'
                Distri_old = torch.distributions.Beta(Distri_old_alpha[Ind_aux_b], Distri_old_beta[Ind_aux_b])
                Kl_loss = torch.distributions.kl_divergence(Distri_old, Distri_new).mean()

                Joint_loss = Kl_loss + Aux_loss
                self.A_optimizer.zero_grad()
                Joint_loss.backward()
                self.A_optimizer.step()

                'Value_loss'
                Value = self.Critic(Aux_s[Ind_aux_b])
                C_loss = 0.5 * F.mse_loss(Value, Aux_V[Ind_aux_b])
                self.C_optimizer.zero_grad()
                C_loss.backward()
                self.C_optimizer.step()

    def save(self, EnvName, timestep):
        torch.save(self.Actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName, timestep))
        torch.save(self.Critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName, timestep))

    def load(self, EnvName, timestep):
        self.Actor.load_state_dict(
            torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=device))
        self.Critic.load_state_dict(
            torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep), map_location=device))


class traj_record(object):
    def __init__(self, T_horizon, state_n, action_n):
        self.s = np.zeros((T_horizon, state_n), dtype=np.float32)
        self.s_ = np.zeros((T_horizon, state_n), dtype=np.float32)
        self.a = np.zeros((T_horizon, action_n), dtype=np.float32)
        self.r = np.zeros((T_horizon, 1), dtype=np.float32)
        self.done = np.zeros((T_horizon, 1), dtype=np.bool_)
        self.dw = np.zeros((T_horizon, 1), dtype=np.bool_)
        self.prob = np.zeros((T_horizon, action_n), dtype=np.float32)

    def add(self, s, a, r, s_, done, Ind, prob_a, dw):
        self.s[Ind] = s
        self.a[Ind] = a
        self.r[Ind] = r
        self.s_[Ind] = s_
        self.done[Ind] = done
        self.prob[Ind] = prob_a
        self.dw[Ind] = dw

    def read(self):
        return (
            torch.FloatTensor(self.s).to(device),
            torch.FloatTensor(self.a).to(device),
            torch.FloatTensor(self.r).to(device),
            torch.FloatTensor(self.s_).to(device),
            torch.FloatTensor(self.prob).to(device),
            torch.BoolTensor(self.done).to(device),
            torch.BoolTensor(self.dw).to(device)
        )
