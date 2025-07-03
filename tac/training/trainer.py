import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
from tqdm import tqdm
from .critic import Critic
from .value_net import ValueNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, state_dim, action_dim, state_mean,state_std, alpha, crtic_lr, loss_fn=None,scheduler=None, eval_fns=None,mode=None):

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

        self.discount = 0.99    # Q-learning discount factor
        self.tau = 0.005         # soft update factor
        
        self.value_net = ValueNet(state_dim).to(device)
        self.value_net_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=crtic_lr)

        # Algorithm 1, line11 : Set hyperparameter alpha 0.9 ~ 2
        self.alpha = alpha
        self.beta = 1.0                # IQL beta值 [1.0,5.0] 默认3.0稳健
        
        self.mode = mode

        self.start_time = time.time()
        print("model train mode: ", self.mode)

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        q_losses = []
        policy_losses = []
        value_losses = []
        cql_alphas = []  # 新增：记录CQL权重变化
        
        logs = dict()

        train_start = time.time()

        self.actor.train()  # 设置模型为训练模式
        print("num_steps",num_steps)
        adv_mean = None
        cql_alpha_current = None
        for _ in tqdm(range(num_steps), desc="train progress"):
            self.total_it += 1
            if self.mode == 'tacr':
                q_loss,policy_loss,q_mean,bc_loss,value_loss = self.train_step(self.total_it)
            elif self.mode == 'cql':
                q_loss,policy_loss,q_mean,bc_loss,cql_alpha_current = self.train_step_cql(self.total_it)
                value_loss = None
                cql_alphas.append(cql_alpha_current)
            elif self.mode == 'iql':
                q_loss,policy_loss,q_mean,adv_mean,value_loss = self.train_step_iql(self.total_it)
            q_losses.append(q_loss)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss if value_loss is not None else 0)

        logs['iter_time'] = time.time() - train_start
        logs['total_time'] = time.time() - self.start_time
        logs['q_loss'] = np.mean(q_losses)
        logs['policy_loss'] = np.mean(policy_losses)
        if q_mean is not None:
            logs['q_mean'] = np.mean(q_mean)
        if bc_loss is not None:
            logs['bc_loss'] = np.mean(bc_loss)
        if adv_mean is not None:
            logs['adv_mean'] = np.mean(adv_mean)
        if value_loss is not None:
            logs['value_loss'] = np.mean(value_losses)
        # 新增：CQL相关日志
        if cql_alphas:
            logs['cql_alpha'] = np.mean(cql_alphas)
            logs['max_cql_alpha'] = np.max(cql_alphas)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k:15s}: {v}')
            
            # 特别显示CQL训练进度
            if self.mode == 'cql' and cql_alpha_current is not None:
                progress_percent = min(100, (self.total_it / 50000) * 100)
                print(f'{"CQL Progress":15s}: {progress_percent:.1f}% (Alpha: {cql_alpha_current:.4f})')

        return logs

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.optimizer.state_dict(), filename + "_actor_optimizer")

