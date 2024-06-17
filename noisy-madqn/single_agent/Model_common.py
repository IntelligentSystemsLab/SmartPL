import torch as th
from torch import nn


class ActorNetwork(nn.Module):
    """
    A network for actor
    """

    def __init__(self, state_dim, hidden_size, output_size, output_act):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # activation function for the output
        self.output_act = output_act

        # -----Noise DQN----------------#
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_mean = th.nn.Parameter(th.randn(hidden_size, output_size))
        self.weight_std = th.nn.Parameter(th.randn(hidden_size, output_size))

        self.bias_mean = th.nn.Parameter(th.randn(output_size))
        self.bias_std = th.nn.Parameter(th.randn(output_size))
        # -----Noise DQN----------------#

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        # out = self.output_act(self.fc3(out))
        # return out
        # -----Noise DQN----------------#
        #正态分布投影,获取输出层的参数
        weight = self.weight_mean + th.randn(
            self.hidden_size, self.output_size).to('cuda') * self.weight_std
        bias = self.bias_mean + th.randn(
            self.output_size).to('cuda') * self.bias_std
        #运行模式下不需要随机性
        if not self.training:
            weight = self.weight_mean
            bias = self.bias_mean

        #计算输出
        return out.matmul(weight) + bias


class CriticNetwork(nn.Module):
    """
    A network for critic
    """

    def __init__(self, state_dim, action_dim, hidden_size, output_size=1):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def __call__(self, state, action):
        out = nn.functional.relu(self.fc1(state))
        out = th.cat([out, action], 1)
        out = nn.functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class ActorCriticNetwork(nn.Module):
    """
    An actor-critic network that shared lower-layer representations but
    have distinct output layers
    """

    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_size,
                 actor_output_act,
                 critic_output_size=1):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.actor_linear = nn.Linear(hidden_size, action_dim)
        self.critic_linear = nn.Linear(hidden_size, critic_output_size)
        self.actor_output_act = actor_output_act

    def __call__(self, state):
        out = nn.functional.relu(self.fc1(state))
        out = nn.functional.relu(self.fc2(out))
        act = self.actor_output_act(self.actor_linear(out))
        val = self.critic_linear(out)
        return act, val
