import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import numpy as np
import copy as cp
from maddpg.buffer import ReplayBuffer


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class q_fun(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU, hiden_sizes=[256,256]):
        super().__init__()
        sizes = [input_dim] + hiden_sizes + [output_dim]
        self.q_fun = mlp(sizes=sizes, activation=activation)

    def forward(self, obs, act):
        q = self.q_fun(torch.cat([obs, act], dim=-1))  # concat the given tensor in given dimension
        # q is has the dimension of (num,1)
        return torch.squeeze(q, -1)


class p_fun(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU, hiden_sizes=[256,256]):
        super().__init__()
        sizes = [input_dim] + hiden_sizes + [output_dim]
        self.p_fun = mlp(sizes=sizes, activation=activation,output_activation = nn.Tanh)

    def forward(self, obs):
        act = self.p_fun(obs)
        return act


class MlpActorCritic(nn.Module):
    def __init__(self, obs_dim_n, act_dim_n, agent_index):
        super().__init__()
        input_dim = np.sum(obs_dim_n) + np.sum(act_dim_n)
        self.q = q_fun(input_dim=input_dim, output_dim=1)
        self.pi = p_fun(input_dim=obs_dim_n[agent_index], output_dim=act_dim_n[agent_index])

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs)

    def q_val(self, obs_n, act_n):
        with torch.no_grad():
            return self.q(obs_n, act_n)


class MaddpgTrainer():  # the obsdim_n and actdim_n has the form of [3,3,3,...]
    def __init__(self, obs_dim_n, agent_index, act_dim_n, env, store_size=int(1e6), q_lr=1e-2, pi_lr=1e-2, gammma=0.95,
                 polyak=0.99):
        self.polyak = polyak
        self.obs_dim, self.act_dim = int(obs_dim_n[agent_index]), int(act_dim_n[agent_index])
        self.agent_index = agent_index
        self.ac = MlpActorCritic(obs_dim_n=obs_dim_n, act_dim_n=act_dim_n,agent_index=agent_index)  # assume
        # all the agent have the same action space
        self.ac_targ = cp.deepcopy(self.ac)
        for parameter in self.ac_targ.parameters():
            parameter.requires_grad = False
        self.q_optimizer = torch.optim.Adam(self.ac.q.parameters(), lr=q_lr)
        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.replay_buffer = ReplayBuffer(obs_dim=self.obs_dim, act_dim=self.act_dim, size=store_size)
        self.gamma = gammma

    def action(self, obs, act_limit=None, noise=None):
        obs = torch.as_tensor(obs,dtype=torch.float32)
        if noise is not None:
            act= self.ac.act(obs).numpy() + noise * np.random.randn()
        else:
            act=self.ac.act(obs).numpy()
        if act_limit is not None:
            return np.clip(act,-act_limit,act_limit)
        else:
            return act

    def store(self, obs, act, obs2, rew, done):
        self.replay_buffer.store(obs, act, rew, obs2, done)

    def update(self, agents):
        if self.replay_buffer.size < 1024:
            return
        obs_n, act_n, obs2_n, rew_n, done_n = [], [], [], [], []
        index = np.random.randint(0,self.replay_buffer.size,size=1024)
        for agent in agents:
            obs_n.append(agent.replay_buffer.sample_batch(idxs=index)['obs'])
            act_n.append(agent.replay_buffer.sample_batch(idxs=index)['act'])
            obs2_n.append(agent.replay_buffer.sample_batch(idxs=index)['obs2'])
            rew_n.append(agent.replay_buffer.sample_batch(idxs=index)['rew'])
            done_n.append(agent.replay_buffer.sample_batch(idxs=index)['done'])
        obs = self.replay_buffer.sample_batch(index)['obs']
        obs2 = self.replay_buffer.sample_batch(index)['obs2']
        act = self.replay_buffer.sample_batch(index)['act']
        rew = self.replay_buffer.sample_batch(index)['rew']
        done = self.replay_buffer.sample_batch(index)['done']

        # compute the q loss and update the q function
        act2 = []
        self.q_optimizer.zero_grad()

        for i, agent in enumerate(agents):
            act2.append(agent.ac_targ.act(obs2_n[i]))
        target = rew + self.gamma * (1 - done) * self.ac_targ.q_val(torch.cat(obs2_n, dim=-1), torch.cat(act2, dim=-1))

        bellman_error = target - self.ac.q(torch.cat(obs_n, dim=-1), torch.cat(act_n, dim=-1))


        loss_q = (bellman_error ** 2).mean()


        loss_q.backward()
        self.q_optimizer.step()

        # compute the pi loss and update the pi function
        current_act = []
        for i, agent in enumerate(agents):
            if agent is self:
                current_act.append(self.ac.pi(obs_n[i]))  # keep the graindient of the current policy
            else:
                current_act.append(agent.ac.act(obs_n[i]))
        for parameter in self.ac.q.parameters():
            parameter.requires_grad = False
        self.pi_optimizer.zero_grad()
        cur_q = self.ac.q(torch.cat(obs_n, dim=-1), torch.cat(current_act, dim=-1))
        loss_pi = -cur_q.mean()
        loss_pi.backward()
        self.pi_optimizer.step()

        for parameter in self.ac.q.parameters():
            parameter.requires_grad = True

        # update the target net parameters
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        return loss_q.detach().numpy()


# ac = MlpActorCritic([3], [3], 0)
#
# obs = np.ones((10, 3))
# act = np.ones((10, 3))
# obs = torch.as_tensor(obs, dtype=torch.float32)
# act = torch.as_tensor(act, dtype=torch.float32)
# for i,p in enumerate(ac.parameters()):
#     print(i,p.data)
