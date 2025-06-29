import torch
import torch.nn.functional as F
from tac.training.trainer import Trainer


class SequenceTrainer(Trainer):
    
    # 原始TACR的训练方法
    def train_step(self,step):
        action_dim = 3
        # Algorithm 1, line6 : Sample a random minibatch
        states, actions, rewards, dones, next_state, next_actions, next_rewards, \
        timesteps, next_timesteps, attention_mask = self.get_batch(self.batch_size)

        # # Algorithm 1, line8 : Predict a action
        state_preds, action_preds, reward_preds = self.actor.forward(
            states, actions, rewards, timesteps, attention_mask=attention_mask,
        )

        next_state_preds, next_action_preds, next_reward_preds = self.actor_target.forward(
            next_state, next_actions, next_rewards, next_timesteps, attention_mask=attention_mask,
        )

        states = states.reshape(-1, self.state_dim)[attention_mask.reshape(-1) > 0]
        next_state = next_state.reshape(-1, self.state_dim)[attention_mask.reshape(-1) > 0]
        rewards = rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        action_sample = actions.reshape(-1, self.action_dim)[attention_mask.reshape(-1) > 0]
        action_preds = action_preds.reshape(-1, self.action_dim)[attention_mask.reshape(-1) > 0]
        next_action_preds = next_action_preds.reshape(-1, self.action_dim)[attention_mask.reshape(-1) > 0]
        dones = dones.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        # next_action_preds 转化为one-hot
        next_action_argmax = next_action_preds.argmax(dim=1)
        next_action_one_hot = torch.eye(action_dim).to(next_action_preds.device)[next_action_argmax]

        # Algorithm 1, line9, line10
        # Compute the target Q value
        target_Q = self.critic_target(next_state, next_action_one_hot)
        target_Q = rewards + (1 - dones) * self.discount * target_Q
        target_Q = target_Q.detach()
        
        # Get current Q estimates
        current_Q = self.critic(states, action_sample)
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_argmax = action_preds.argmax(dim=1)
        action_one_hot = torch.eye(action_dim).to(action_preds.device)[action_argmax]
        # Algorithm 1, line11, line12 : Set lambda and Compute actor loss
        Q = self.critic(states, action_one_hot)
        lmbda = self.alpha / (Q.abs().mean().detach() + 1e-6)
        bc_loss = F.cross_entropy(action_preds, action_sample.argmax(dim=-1))
        actor_loss = -lmbda * Q.mean() + bc_loss
        
        

        # Optimize the actor
        self.optimizer.zero_grad()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .25)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # # Algorithm 1, line13 : Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # 返回 q_loss,policy_loss,value_loss(None)
        return (critic_loss.detach().cpu().item(),
                actor_loss.detach().cpu().item(),
                Q.mean().detach().cpu().item(),
                bc_loss.detach().cpu().item(),
                None # value_loss
                )


    # CQL 的方法
    def train_step_cql(self,step):
        
        num_actions = 3
        # Algorithm 1, line6 : Sample a random minibatch
        states, actions, rewards, dones, next_state, next_actions, next_rewards, \
        timesteps, next_timesteps, attention_mask = self.get_batch(self.batch_size)

        # # Algorithm 1, line8 : Predict a action
        state_preds, action_preds, reward_preds = self.actor.forward(
            states, actions, rewards, timesteps, attention_mask=attention_mask,
        )

        # actor_target 是actor的一个副本
        next_state_preds, next_action_preds, next_reward_preds = self.actor_target.forward(
            next_state, next_actions, next_rewards, next_timesteps, attention_mask=attention_mask,
        )

        states = states.reshape(-1, self.state_dim)[attention_mask.reshape(-1) > 0]
        next_state = next_state.reshape(-1, self.state_dim)[attention_mask.reshape(-1) > 0]
        rewards = rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        # 当前的轨迹中的动作,用于critic的输入
        action_sample = actions.reshape(-1, self.action_dim)[attention_mask.reshape(-1) > 0]
        
        action_preds = action_preds.reshape(-1, self.action_dim)[attention_mask.reshape(-1) > 0]
        next_action_preds = next_action_preds.reshape(-1, self.action_dim)[attention_mask.reshape(-1) > 0]
        dones = dones.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        # next_action_preds 转化为one-hot
        next_action_argmax = next_action_preds.argmax(dim=1)
        next_action_one_hot = torch.eye(num_actions).to(next_action_preds.device)[next_action_argmax]
        
        # print("dones",dones)

        # Algorithm 1, line9, line10
        # Compute the target Q value
        target_Q = self.critic_target(next_state, next_action_one_hot)
        target_Q = rewards + (1 - dones) * self.discount * target_Q
        target_Q = target_Q.detach()

        # Get current Q estimates
        current_Q = self.critic(states, action_sample)
        
        
        
        # CQL 正则项：最小化随机采样动作的 Q 值，同时最大化数据集中动作的 Q 值
        # 使用 CQL(H) 变体，通过 log-sum-exp 实现
        # 计算 CQL 正则项
        batch_size = states.shape[0]
        
        # num_random = 10  # 随机动作的倍数
        # random_actions = torch.FloatTensor(batch_size * num_random, self.action_dim).uniform_(-1, 1).to(states.device)
        # repeated_states = states.unsqueeze(1).repeat(1, num_random, 1).reshape(batch_size * num_random, -1)
        # random_q = self.critic(repeated_states, random_actions)
        # random_q = random_q.reshape(batch_size, num_random, 1)
        
        # 离散动作使用全部空间
        
        all_actions = torch.eye(num_actions).to(states.device)
        repeated_states = states.unsqueeze(1).repeat(1, num_actions, 1).reshape(batch_size * num_actions, -1)
        repeated_actions = all_actions.unsqueeze(0).repeat(batch_size, 1, 1).reshape(batch_size * num_actions, -1)
        all_q = self.critic(repeated_states, repeated_actions).reshape(batch_size, num_actions)
        logsumexp_q = torch.logsumexp(all_q, dim=1, keepdim=True)
        cql_regularizer = (logsumexp_q - current_Q).mean()

        
        # data_q = current_Q.unsqueeze(1)
        
        # action_preds 转化为one-hot
        action_argmax = action_preds.argmax(dim=1)
        action_one_hot = torch.eye(num_actions).to(action_preds.device)[action_argmax]
        
        Q = self.critic(states, action_one_hot)
        # policy_q = Q.unsqueeze(1)
        
        # 最终的 critic 损失 = 标准 TD 误差 + CQL 正则项
        critic_loss = F.mse_loss(current_Q, target_Q) + self.alpha * cql_regularizer

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 当前预测动作的Q值
        new_Q = self.critic(states, action_one_hot)

        # Algorithm 1, line11, line12 : 计算 actor loss
        # 在 CQL 方法中，我们仍然使用类似的 actor 更新
        
        # lmbda = self.alpha / new_Q.abs().mean().detach()
        
        # 离散动作使用交叉熵损失
        bc_loss = F.cross_entropy(action_preds, action_sample.argmax(dim=-1))
        # self.alpha = 0.0

        actor_loss = -self.alpha * new_Q.mean() + bc_loss

        # Optimize the actor 训练主网络
        self.optimizer.zero_grad()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .25)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        # # Algorithm 1, line13 : Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return (critic_loss.detach().cpu().item(),
                actor_loss.detach().cpu().item(),
                new_Q.mean().detach().cpu().item(),
                bc_loss.detach().cpu().item(),
                None)
    
    # IQL 的方法
    def train_step_iql(self,step):
        # Algorithm 1, line6 : Sample a random minibatch
        states, actions, rewards, dones, next_state, next_actions, next_rewards, \
        timesteps, next_timesteps, attention_mask = self.get_batch(self.batch_size)

        # # Algorithm 1, line8 : Predict a action
        action_preds, log_probs, action_mean, alpha = self.actor.forward_dist(
            states, actions, rewards, timesteps, attention_mask=attention_mask,
        )
        
        # action_preds已经在forward_dist方法中通过sigmoid限制在了[0,1]范围内

        states = states.reshape(-1, self.state_dim)[attention_mask.reshape(-1) > 0]
        rewards = rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        # 当前的轨迹中的动作,用于critic的输入
        action_sample = actions.reshape(-1, self.action_dim)[attention_mask.reshape(-1) > 0]
        
        action_preds = action_preds.reshape(-1, self.action_dim)[attention_mask.reshape(-1) > 0]
        
        dones = dones.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        
        next_state = next_state.reshape(-1, self.state_dim)[attention_mask.reshape(-1) > 0]
        
        
        # 1. 首先更新V(s)
        v = self.value_net(states)

        # Get current Q estimates
        with torch.no_grad():
            current_Q = self.critic(states, action_sample)
        
        v_loss = F.mse_loss(v, current_Q)
        
        self.value_net_optimizer.zero_grad()
        v_loss.backward()
        self.value_net_optimizer.step()
        

        batch_size = states.shape[0]
        

        # 2. 更新Q网络
        with torch.no_grad():
            next_v = self.value_net(next_state)
            target = rewards + self.discount * next_v * (1 - dones)
        
        q_pred = self.critic(states, action_sample)

        q_loss = F.mse_loss(q_pred, target)

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()


        
        # 3. 更新actor网络
        with torch.no_grad():
            new_Q = self.critic(states, action_preds)
            new_v = self.value_net(states)
            adv = new_Q - new_v
            # 标准化 advantage
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
            exp_adv = torch.exp(adv / self.beta).clamp(max=100)
            
        
        # print("log_probs shape",log_probs.shape)
        # print("exp_adv shape",exp_adv.shape)
        log_probs = log_probs.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
        actor_loss = -(exp_adv * log_probs).mean()
        
        
        # if step % 1000 == 0:
        #     print("new_Q mean       ",new_Q.mean().item(),"min",new_Q.min().item(),"max",new_Q.max().item())
        #     print("new_v mean       ",new_v.mean().item(),"min",new_v.min().item(),"max",new_v.max().item())
        #     print("action_preds mean",action_preds.mean().item(),"min",action_preds.min().item(),"max",action_preds.max().item())
        #     print("log_probs mean   ",log_probs.mean().item(),"min",log_probs.min().item(),"max",log_probs.max().item())
        #     print("action_mean mean ",action_mean.mean().item(),"min",action_mean.min().item(),"max",action_mean.max().item())
        #     print("alpha mean       ",alpha.mean().item(),"min",alpha.min().item(),"max",alpha.max().item())
        #     print("adv mean         ",adv.mean().item(),"min",adv.min().item(),"max",adv.max().item())
        #     print("exp_adv mean     ",exp_adv.mean().item(),"min",exp_adv.min().item(),"max",exp_adv.max().item())
        #     print("--[actor_loss ]--",actor_loss.mean().item(),"min",actor_loss.min().item(),"max",actor_loss.max().item())


        # Optimize the actor 训练主网络
        self.optimizer.zero_grad()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .25)
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()


        return (q_loss.detach().cpu().item(),
                actor_loss.detach().cpu().item(),
                new_Q.mean().detach().cpu().item(),
                exp_adv.mean().detach().cpu().item(),
                v_loss.detach().cpu().item()
                )

