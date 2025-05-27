import torch
import torch.nn.functional as F
from tac.training.trainer import Trainer


class SequenceTrainer(Trainer):

    def train_step(self):
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
        
        print("dones",dones)

        # Algorithm 1, line9, line10
        # Compute the target Q value
        target_Q = self.critic_target(next_state, next_action_preds)
        target_Q = rewards + (dones * self.discount * target_Q).detach()
        # Get current Q estimates
        current_Q = self.critic(states, action_sample)
        
        # 当前预测动作的Q值
        Q = self.critic(states, action_preds)
        
        # CQL 正则项：最小化随机采样动作的 Q 值，同时最大化数据集中动作的 Q 值
        # 使用 CQL(H) 变体，通过 log-sum-exp 实现
        # 计算 CQL 正则项
        batch_size = states.shape[0]
        
        num_random = 10  # 随机动作的倍数
        random_actions = torch.FloatTensor(batch_size * num_random, self.action_dim).uniform_(-1, 1).to(states.device)
        repeated_states = states.unsqueeze(1).repeat(1, num_random, 1).reshape(batch_size * num_random, -1)
        random_q = self.critic(repeated_states, random_actions)
        random_q = random_q.reshape(batch_size, num_random, 1)
        
        # log-sum-exp 计算
        cat_q = torch.cat([random_q, Q.unsqueeze(1)], dim=1)
        logsumexp_q = torch.logsumexp(cat_q, dim=1, keepdim=True)
        
        # CQL 正则项 = alpha * (logsumexp_q - q_data)
        cql_regularizer = (logsumexp_q - current_Q).mean()
        cql_alpha = self.alpha  # 使用已有的 alpha 参数
        
        # 最终的 critic 损失 = 标准 TD 误差 + CQL 正则项
        critic_loss = F.mse_loss(current_Q, target_Q) + cql_alpha * cql_regularizer

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Algorithm 1, line11, line12 : 计算 actor loss
        # 在 CQL 方法中，我们仍然使用类似的 actor 更新
        
        lmbda = self.alpha / Q.abs().mean().detach()
        bc_loss = F.mse_loss(action_preds, action_sample)
        actor_loss = -lmbda * Q.mean() + bc_loss

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

        return actor_loss.detach().cpu().item()

