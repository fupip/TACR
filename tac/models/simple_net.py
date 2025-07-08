import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """
    简单的MLP网络,用于替代GPT2Model
    保持与GPT2Model相同的接口
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embd
        self.num_layers = getattr(config, 'num_layers', 3)
        
        # 构建MLP层
        layers = []
        for i in range(self.num_layers):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
        
        self.mlp = nn.Sequential(*layers)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
    def forward(self, inputs_embeds, attention_mask=None):
        """
        前向传播保持与GPT2Model相同的接口
        
        Args:
            inputs_embeds: 输入嵌入 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            dict: 包含'last_hidden_state'的字典
        """
        batch_size, seq_len, hidden_size = inputs_embeds.shape
        
        # 应用attention mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds * mask_expanded.float()
        
        # 通过MLP处理
        hidden_states = self.mlp(inputs_embeds)
        
        # 层归一化
        hidden_states = self.layer_norm(hidden_states)
        
        # 返回与GPT2Model相同格式的输出
        return {
            'last_hidden_state': hidden_states
        } 