# 2021-Phasic Policy Gradient （PPG-discrete）
import gym
import torch
from PPG_discrete import PPG_Agent, traj_memory
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def str2bool(V):
    if isinstance(V, bool):
        return V
    elif V.lower in ('yes', 'true', 't', 'y'):
        return True
    elif V.lower in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def eval_func(env_eval, model, eval_seed, e_turns):
    score = 0
    for j in range(e_turns):
        s, _ = env_eval.reset()
        done, er_r = False, 0
        while not done:
            a, _ = model.action_select(s, True)
            s_, r, dw, tr, _ = env_eval.step(a)
            done = (dw or tr)

            er_r += r
            s = s_
        score += er_r
    return score / e_turns


def show_func(env_eval, model, eval_seed, e_turns):
    score = 0
    for j in range(e_turns):
        s, _ = env_eval.reset()
        done, er_r = False, 0
        eval_seed += 1
        while not done:
            a, _ = model.action_select(s, True)
            s_, r, dw, tr, _ = env_eval.step(a)
            done = (dw or tr)
            score += r
            s = s_
    return score / e_turns


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=1, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--show', type=str2bool, default=False, help='show or not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=300000, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--T_horizon', type=int, default=1042, help='lenth of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=1e6, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--epochs_aux', type=int, default=6, help='aux update times')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=64, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=64, help='lenth of sliced trajectory')
parser.add_argument('--entropy_coef', type=float, default=0, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
parser.add_argument('--N_pi', type=int, default=int(32), help='freq of aux update')
opt = parser.parse_args()
print(opt)


def main():
    if opt.EnvIdex == 0:
        Bench = 'CartPole-v1'
        BName = 'CPV0'
        opt.env_dw = True
    else:
        Bench = 'LunarLander-v2'
        BName = 'LLV2'
        opt.env_dw = True
    env_train = gym.make(Bench, render_mode="human" if opt.render else None)
    env_eval = gym.make(Bench, render_mode="human" if opt.render else None)
    opt.state_dim = env_eval.observation_space.shape[0]
    opt.action_dim = env_eval.action_space.n
    opt.max_e_steps = env_eval._max_episode_steps
    opt.algo_name = 'PPG'

    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Algorithm:', opt.algo_name, '  Env:', BName, '  state_dim:', opt.state_dim,
          '  action_dim:', opt.action_dim, '  Random Seed:', opt.seed, '  max_e_steps:', opt.max_e_steps, '\n')

    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}_{}'.format(opt.algo_name, BName) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    if not os.path.exists('model'): os.mkdir('model')
    model = PPG_Agent(opt)
    if opt.Loadmodel: model.load(BName, opt.ModelIdex)
    traj = traj_memory(opt.T_horizon, opt.state_dim)

    if opt.render:
        eval_seed = env_seed
        while True:
            score = eval_func(env_eval, model, eval_seed, 3)
            print(f'Env:{BName}, seed:{eval_seed}, Episode Reward:{score}')
            eval_seed = 1
    else:
        traj_len = 0
        total_steps = 0
        Aux_count = 0
        while total_steps <= opt.Max_train_steps:
            env_seed += 1
            s, _ = env_train.reset(seed=env_seed)
            done, steps = False, 0
            while not done:
                a, prob_a = model.action_select(s, False)
                s_, r, dw, tr, _ = env_train.step(a)
                if opt.EnvIdex == 1:
                    if r <= -100:
                        r = -10
                done = (dw or tr)
                total_steps += 1
                traj.add(s, a, prob_a, r, s_, done, traj_len)
                s = s_
                traj_len += 1

                if traj_len == opt.T_horizon:
                    model.train(traj)
                    traj_len = 0
                    Aux_count += 1

                    if Aux_count % opt.N_pi == 0:
                        model.aux_train()

                if total_steps % opt.eval_interval == 0 or total_steps == 1:
                    score = eval_func(env_eval, model, env_seed, 3)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                    print('EnvName:', BName, 'seed:', env_seed,
                          'steps: {}k'.format(int(total_steps / 1000)), 'score:', score)

                if total_steps % opt.save_interval == 0:
                    model.save(BName, total_steps)

    env_show = gym.make(Bench, render_mode="human")
    score = show_func(env_show, model, 0, 10)
    print('EnvName:', BName, 'seed:', 0, 'score:', score)
    env_show.close()
    env_train.close()
    env_eval.close()


if __name__ == '__main__':
    main()
