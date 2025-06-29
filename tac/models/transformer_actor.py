import numpy as np
import torch
import torch.nn as nn
import transformers
from tac.models.model import TrajectoryModel
from tac.models.trajectory_gpt2 import GPT2Model
import torch.nn.functional as F

class TransformerActor(TrajectoryModel):

    """
    This model uses GPT to model (reward_1, state_1, action_1, reward_2, state_2, ...)
    """

    def __init__(
            self,
            state_dim,
            act_dim,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_softmax=True,
            **kwargs
    ):
        super().__init__(state_dim, act_dim, max_length=max_length)

        self.hidden_size = hidden_size
        config = transformers.GPT2Config(
            vocab_size=1,  # doesn't matter -- we don't use the vocab
            n_embd=hidden_size,
            **kwargs
        )

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        self.embed_ln = nn.LayerNorm(hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(hidden_size, self.act_dim)] + ([nn.Softmax(dim=2)] if action_softmax else []))
        )
        self.predict_return = torch.nn.Linear(hidden_size, 1)
        
        self.action_head_alpha = nn.Linear(hidden_size, self.act_dim)

    def forward(self, states, actions, rewards, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # Subsection 4.1, (9) : Embeddings of MDP
        # embed each modality with a different head
        # print("states.shape",states.shape)
        # print("actions.shape",actions.shape)
        # print("rewards.shape",rewards.shape)
        # print("timesteps.shape",timesteps.shape)
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(rewards)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Algorithm 1, line7 : Stack MDP elements
        # this makes the sequence look like (r_1, s_1, a_1, r_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        # print("returns_embeddings.shape",returns_embeddings.shape)
        # print("state_embeddings.shape",state_embeddings.shape)
        # print("action_embeddings.shape",action_embeddings.shape)
        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        # print("stacked_inputs.shape",stacked_inputs.shape)
        
        stacked_inputs = self.embed_ln(stacked_inputs)
        # print("embed_ln stacked_inputs.shape",stacked_inputs.shape)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        return_preds = self.predict_return(x[:,2])  # Predict next return given state and action (we don't use this)
        state_preds = self.predict_state(x[:,2])    # Predict next state given state and action (we don't use this)
        action_preds = self.predict_action(x[:,1])  # Algorithm 1, line8 : Predict next action given state
        # print("action_preds.shape",action_preds.shape)
        # print("return_preds.shape",return_preds.shape)
        # print("state_preds.shape",state_preds.shape)
        # print("--------------------------------")
        return state_preds, action_preds, return_preds
    
    def predict_action_dist(self, h):
        # h: [batch, seq, hidden_size]
        alpha_raw = self.action_head_alpha(h)  # 假设你有一个Linear层输出30维
        # 更严格地限制alpha的值，防止Dirichlet分布参数过大
        alpha = torch.clamp(F.softplus(alpha_raw) + 0.0001, min=1.0, max=8.0)
        return alpha    
    
    # 预测动作分布 为IQL算法使用
    def forward_dist(self, states, actions, rewards, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        

        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        # Subsection 4.1, (9) : Embeddings of MDP
        # embed each modality with a different head
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(rewards)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # Algorithm 1, line7 : Stack MDP elements
        # this makes the sequence look like (r_1, s_1, a_1, r_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions

        stacked_inputs = torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # reshape x so that the second dimension corresponds to the original
        # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
        x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

        # get predictions
        
        # mu, log_std = self.predict_action_dist(x[:,1])
        # std = log_std.exp()
        # dist = torch.distributions.Normal(mu, std)
        # raw_actions = dist.rsample()         # 采样原始动作
        # log_probs = dist.log_prob(raw_actions).sum(-1) # 计算原始动作的log_prob
        # # 将动作限制在[0,1]范围内，并且各维度和为1
        # action_preds = F.softmax(raw_actions, dim=-1)  # 使用softmax替代sigmoid
        
        # # 对mu也应用softmax
        # action_mean = F.softmax(mu, dim=-1)
        
        
        alpha = self.predict_action_dist(x[:,1])                  # [batch, seq, 30]
        dist = torch.distributions.Dirichlet(alpha)
        action_preds = dist.rsample()                          # [batch, seq, 30]，每行和为1
        log_probs = dist.log_prob(action_preds)                   # [batch, seq]


        action_mean = alpha / alpha.sum(dim=-1, keepdim=True)     # Dirichlet均值，[batch, seq, 30]
        
        
        # print("action_preds mean",action_preds.mean(),"min",action_preds.min(),"max",action_preds.max())
        # print("log_probs mean",log_probs.mean(),"min",log_probs.min(),"max",log_probs.max())
        # print("action_mean mean",action_mean.mean(),"min",action_mean.min(),"max",action_mean.max())
        # print('alpha mean', alpha.mean(), 'min', alpha.min(), 'max', alpha.max())


        return action_preds, log_probs, action_mean, alpha

    def get_action(self, states, actions, rewards,  timesteps, **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        rewards = rewards.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            rewards = rewards[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)

            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], self.state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            rewards = torch.cat(
                [torch.zeros((rewards.shape[0], self.max_length-rewards.shape[1], 1), device=rewards.device), rewards],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        # _, action_preds, return_preds = self.forward(
        #     states, actions, rewards, timesteps, attention_mask=attention_mask, **kwargs)
        
        # 使用IQL算法
        action_preds, log_probs, action_mean, alpha = self.forward_dist(
            states, actions, rewards, timesteps, attention_mask=attention_mask, **kwargs)
        
        # print("action_preds_sample mean",action_preds_sample.mean().item(),"min",action_preds_sample.min().item(),"max",action_preds_sample.max().item())
        # print("action_preds mean",action_preds.mean().item(),"min",action_preds.min().item(),"max",action_preds.max().item())
        # print("log_probs mean",log_probs.mean().item(),"min",log_probs.min().item(),"max",log_probs.max().item())
        # print("alpha mean",alpha.mean().item(),"min",alpha.min().item(),"max",alpha.max().item())

        result = action_preds[0,-1]
        # print("result: ", result)
        
        argmax = torch.argmax(action_preds, dim=-1)
        # print("argmax: ", argmax)

        
        return result
