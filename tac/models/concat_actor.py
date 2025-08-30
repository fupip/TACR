import torch
import torch.nn as nn
from tac.models.simple_net import SimpleNet

class ConcatenatedTransformerActor(nn.Module):
    """
    使用拼接方案的TransformerActor：将reward、state、action拼接为单个向量
    """
    
    def __init__(self, state_dim, act_dim, hidden_size=64, max_length=20, max_ep_len=4096):
        super().__init__()
        
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # 拼接后的输入维度：状态 + 奖励(1维) + 动作
        self.input_dim = state_dim + 1 + act_dim
        
        # 创建SimpleNet的配置
        class SimpleConfig:
            def __init__(self):
                self.n_embd = hidden_size
                self.num_layers = 3
        
        config = SimpleConfig()
        
        # 使用SimpleNet作为主干网络
        self.transformer = SimpleNet(config)
        
        # 时间步嵌入
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        
        # 拼接向量的嵌入层
        self.embed_concat = nn.Linear(self.input_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        
        # 预测头
        self.predict_action = nn.Linear(hidden_size, act_dim)
        
    def forward(self, states, actions, rewards, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)
        
        # 将奖励reshape为正确的维度 [batch_size, seq_length, 1]
        if rewards.dim() == 2:
            rewards = rewards.unsqueeze(-1)
        
        # 拼接 states + rewards + actions
        # states: [batch_size, seq_length, state_dim]
        # rewards: [batch_size, seq_length, 1] 
        # actions: [batch_size, seq_length, act_dim]
        concatenated_input = torch.cat([states, rewards, actions], dim=-1)
        # concatenated_input: [batch_size, seq_length, state_dim + 1 + act_dim]
        
        # 嵌入拼接的向量
        embedded_input = self.embed_concat(concatenated_input)
        # embedded_input: [batch_size, seq_length, hidden_size]
        
        # 添加时间嵌入
        time_embeddings = self.embed_timestep(timesteps)
        embedded_input = embedded_input + time_embeddings
        
        # 层归一化
        embedded_input = self.embed_ln(embedded_input)
        
        # 通过Transformer处理
        transformer_outputs = self.transformer(
            inputs_embeds=embedded_input,
            attention_mask=attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        
        # 预测动作
        action_preds = self.predict_action(x)
        
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
        
        # 前向传播
        with torch.no_grad():
            action_preds = self.forward(states, actions, rewards, timesteps, attention_mask=attention_mask)
            
        # 应用softmax进行归一化
        action_preds = torch.softmax(action_preds, dim=-1)
        
        # 验证输出是否为有效概率分布
        result = action_preds[0, -1]
        print(f"ConcatActor - Action output: {result.detach().cpu().numpy()}, sum: {result.sum().item():.6f}")
        
        return result
