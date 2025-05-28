import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
from .critic import Critic
from .value_net import ValueNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, state_dim, action_dim, state_mean,state_std, alpha, crtic_lr, loss_fn=None,scheduler=None, eval_fns=None):

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_mean=state_mean
        self.state_std = state_std
        self.total_it = 0

        # Algorithm 1, line1, line2 : Initialize actor and critic weights
        self.actor = model
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=crtic_lr)

        self.discount = 0.99
        self.tau = 0.005
        
        self.value_net = ValueNet(state_dim).to(device)
        self.value_net_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=crtic_lr)

        # Algorithm 1, line11 : Set hyperparameter alpha 0.9 ~ 2
        self.alpha = alpha
        self.beta = 3.0                # IQL beta值 [1.0,5.0] 默认3.0稳健

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.actor.train()  # 设置模型为训练模式
        print("num_steps",num_steps)
        for _ in range(num_steps):
            self.total_it += 1
            train_loss = self.train_step_iql()
            train_losses.append(train_loss)

        logs['time/training'] = time.time() - train_start
        logs['time/total'] = time.time() - self.start_time
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.optimizer.state_dict(), filename + "_actor_optimizer")

