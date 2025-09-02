import torch
import torch.nn as nn
from tac.models.simple_net import SimpleNet

class SimpleTransformerActor(nn.Module):
    """
    简化版的TransformerActor,使用SimpleNet替代GPT2Model
    """
    
    def __init__(self, state_dim, act_dim, hidden_size=64, max_length=20, max_ep_len=4096):
        super().__init__()
        
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # 创建SimpleNet的配置
        class SimpleConfig:
            def __init__(self):
                self.n_embd = hidden_size
                self.num_layers = 3
        
        config = SimpleConfig()
        
        # 使用SimpleNet作为主干网络
        self.transformer = SimpleNet(config)
        
        # 嵌入层
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # 预测头
        self.predict_action = nn.Linear(hidden_size, act_dim)
        
    def forward(self, states, actions, rewards, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)
        
        # print("simple actor forward: states",states,states.shape)
        # print("simple actor forward: actions",actions,actions.shape)
        # print("simple actor forward: rewards",rewards,rewards.shape)
        # print("simple actor forward: timesteps",timesteps,timesteps.shape)
        
        # 嵌入各个模态
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(rewards)
        time_embeddings = self.embed_timestep(timesteps)
        
        # 添加时间嵌入
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        
        # 堆叠输入序列 (reward, state, action)
        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1
                                    ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        
        # u =10  hidden_size = 64
        # stacked_inputs 的序列结构 (shape: 1, 30, 64)
        #         stacked_inputs = [
        #         emb_r_1,   # position 0  - 第1天奖励
        #         emb_s_1,   # position 1  - 第1天状态  
        #         emb_a_1,   # position 2  - 第1天动作
        #         emb_r_2,   # position 3  - 第2天奖励
        #         emb_s_2,   # position 4  - 第2天状态
        #         emb_a_2,   # position 5  - 第2天动作
        #         emb_r_3,   # position 6  - 第3天奖励
        #         emb_s_3,   # position 7  - 第3天状态
        #         emb_a_3,   # position 8  - 第3天动作
        #         ...
        #         emb_r_10,  # position 27 - 第10天奖励
        #         emb_s_10,  # position 28 - 第10天状态
        #         emb_a_10   # position 29 - 第10天动作
        #       ]
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # 堆叠attention mask
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        
        # 通过SimpleNet处理
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        
        # 重新整形
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        
        # 预测动作（基于状态）
        action_preds = self.predict_action(x[:,1])  # 使用状态位置的输出
        
        return action_preds
    
    def get_action(self, states, actions, rewards, timesteps):
        """预测下一步动作"""
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)
        
        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            rewards = rewards[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]
            
            # 创建attention mask
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), 
                                    torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            
            # 填充序列
            states = torch.cat([torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), 
                                        device=states.device), states], dim=1).to(dtype=torch.float32)
            actions = torch.cat([torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                                        device=actions.device), actions], dim=1).to(dtype=torch.float32)
            rewards = torch.cat([torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1), 
                                        device=rewards.device), rewards], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat([torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), 
                                        device=timesteps.device), timesteps], dim=1).to(dtype=torch.long)
        else:
            attention_mask = None
        
        print("simple actor get_action: states",states,states.shape)  
        # 前向传播
        with torch.no_grad():
            action_preds = self.forward(states, actions, rewards, timesteps, attention_mask=attention_mask)
            
        # 应用softmax进行归一化，确保输出是有效的概率分布
        action_preds = torch.softmax(action_preds, dim=-1)
        
        # 验证输出是否为有效概率分布
        result = action_preds[0, -1]
        print(f"Action output: {result.detach().cpu().numpy()}, sum: {result.sum().item():.6f}")
        
        return result  # 返回最后一个时间步的预测