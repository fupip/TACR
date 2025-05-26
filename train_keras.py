import numpy as np
import wandb
import argparse
import random
import pickle
import pandas as pd
from stock_env.apps import config
from stock_env.allocation.env_portfolio import StockPortfolioEnv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# 设置内存增长模式以避免占用所有GPU内存
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# 创建Transformer Actor模型（keras版本）
class KerasTransformerActor(keras.Model):
    def __init__(
        self,
        state_dim,
        act_dim,
        max_length,
        max_ep_len,
        hidden_size,
        n_layer,
        n_head,
        activation_function='relu',
        dropout_rate=0.1,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        
        # 位置编码层
        self.embed_timestep = keras.layers.Embedding(max_ep_len, hidden_size)
        
        # 状态编码层
        self.embed_state = keras.Sequential([
            keras.layers.Dense(hidden_size, activation=activation_function),
            keras.layers.LayerNormalization()
        ])
        
        # Transformer编码器层
        self.transformer_blocks = []
        for _ in range(n_layer):
            self.transformer_blocks.append(
                TransformerBlock(hidden_size, n_head, hidden_size * 4, dropout_rate)
            )
        
        # 输出层
        self.predict_action = keras.layers.Dense(act_dim)
        
    def call(self, states, timesteps, training=False):
        batch_size = tf.shape(states)[0]
        
        # 创建掩码（用于Transformer的注意力机制）
        mask = 1.0 - tf.linalg.band_part(
            tf.ones((self.max_length, self.max_length)), -1, 0
        )
        mask = tf.reshape(mask, [1, 1, self.max_length, self.max_length])
        
        # 编码状态和时间步
        state_embeddings = self.embed_state(states)
        time_embeddings = self.embed_timestep(timesteps)
        
        # 合并编码
        h = state_embeddings + time_embeddings
        
        # 通过Transformer blocks
        for block in self.transformer_blocks:
            h = block(h, mask, training=training)
            
        # 预测动作
        actions = self.predict_action(h)
        
        return actions

# Transformer Block
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        
    def call(self, inputs, mask, training=False):
        attn_output = self.att(inputs, inputs, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# 序列训练器（Keras版本）
class KerasSequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        critic_optimizer,
        batch_size,
        get_batch,
        action_dim,
        state_dim,
        state_mean,
        state_std,
        alpha=0.9
    ):
        self.model = model
        self.optimizer = optimizer
        self.critic_optimizer = critic_optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.alpha = alpha
        
        # 创建critic模型
        self.critic = self._create_critic_model()
    
    def _create_critic_model(self):
        state_input = keras.layers.Input(shape=(self.state_dim,))
        action_input = keras.layers.Input(shape=(self.action_dim,))
        
        # 合并状态和动作
        x = keras.layers.Concatenate()([state_input, action_input])
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        q_value = keras.layers.Dense(1)(x)
        
        return keras.Model([state_input, action_input], q_value)
    
    def train_iteration(self, num_steps=1000, iter_num=0, print_logs=False):
        logs = {}
        train_loss = []
        train_critic_loss = []
        
        for _ in range(num_steps):
            s, a, r, d, next_s, next_a, next_r, timesteps, n_timesteps, mask = self.get_batch(self.batch_size)
            
            with tf.GradientTape() as tape:
                # 预测动作
                pred_actions = self.model(s, timesteps, training=True)
                # 计算预测动作在掩码下的均方误差损失
                action_mse = tf.reduce_mean(tf.square(pred_actions - a) * mask[..., None])
                loss = action_mse
                
            # 优化actor模型
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            train_loss.append(loss.numpy())
            
            # 训练critic模型（用于TD学习或其他策略优化方法）
            with tf.GradientTape() as critic_tape:
                # 将批次中的状态和动作展平用于critic训练
                flat_states = tf.reshape(s, [-1, self.state_dim])
                flat_actions = tf.reshape(a, [-1, self.action_dim])
                flat_next_states = tf.reshape(next_s, [-1, self.state_dim])
                flat_next_actions = tf.reshape(next_a, [-1, self.action_dim])
                flat_rewards = tf.reshape(r, [-1, 1])
                flat_dones = tf.reshape(d, [-1, 1])
                flat_mask = tf.reshape(mask, [-1, 1])
                
                # 计算当前Q值
                current_q = self.critic([flat_states, flat_actions])
                
                # 计算目标Q值
                target_q = flat_rewards + (1.0 - flat_dones) * self.alpha * self.critic([flat_next_states, flat_next_actions])
                
                # 计算critic损失
                critic_loss = tf.reduce_mean(tf.square(current_q - target_q) * flat_mask)
                
            # 优化critic模型
            critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            train_critic_loss.append(critic_loss.numpy())
            
        logs['train_loss'] = np.mean(train_loss)
        logs['train_critic_loss'] = np.mean(train_critic_loss)
        
        if print_logs:
            print(f"Iteration: {iter_num}, Loss: {logs['train_loss']:.4f}, Critic Loss: {logs['train_critic_loss']:.4f}")
            
        return logs
    
    def save(self, path):
        self.model.save_weights(path)

def main(variant):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)
    
    if device == 'cuda' and not tf.config.list_physical_devices('GPU'):
        print("CUDA requested but not available, using CPU instead.")
        device = 'cpu'

    env_name, dataset = variant['env'], variant['dataset']
    group_name = f'{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    train = pd.read_csv("datasets/"+dataset+"_train.csv", index_col=[0])
    max_ep_len = train.index[-1]

    # 加载次优轨迹
    dataset_path = f'{"trajectory/" + variant["dataset"] + "_traj.pkl"}'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    state_space=trajectories[0]['observations'].shape[1]
    stock_dimension = len(train.tic.unique())

    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    # 设置投资组合分配环境
    env_kwargs = {
        "dataset": dataset,
        "initial_amount": 1000000,
        "transaction_cost": 0.0025,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
    }
    env = StockPortfolioEnv(df=train, **env_kwargs)

    # 设置随机种子
    seed = variant['seed']
    env.seed(seed)
    env.action_space.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 算法1，第3行：设置序列长度u
    u = variant['u']

    model = KerasTransformerActor(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=u,
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        activation_function=variant['activation_function'],
        dropout_rate=variant['dropout'],
    )

    states, traj_lens, returns = [], [], []
    for path in trajectories:
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # 用于输入归一化
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    batch_size = variant['batch_size']

    def get_batch(batch_size=64, max_len=u):
        batch_inds = np.random.choice(
            np.arange(len(traj_lens)),
            size=batch_size,
            replace=True
        )

        s, next_s, next_a, next_r, a, r, d, dd, timesteps, n_timesteps, mask \
            = [], [], [], [], [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(batch_inds[i])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)
            # 从数据集获取序列
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            dd.append(traj['dones'][si:si + max_len].reshape(1, -1, 1))

            if si >= traj['rewards'].shape[0] - u:
                next_s.append(np.append(traj['observations'][si + 1:si + 1 + max_len],
                                        traj['observations'][traj['rewards'].shape[0] - 1]).reshape(1, -1, state_dim))
                next_a.append(np.append(traj['actions'][si + 1:si + 1 + max_len],
                                        traj['actions'][traj['rewards'].shape[0] - 1]).reshape(1, -1, act_dim))
                next_r.append(np.append(traj['rewards'][si + 1:si + 1 + max_len],
                                        np.array([traj['rewards'][traj['rewards'].shape[0] - 1]])).reshape(1, -1, 1))
            else:
                next_s.append(traj['observations'][si + 1:si + 1 + max_len].reshape(1, -1, state_dim))
                next_a.append(traj['actions'][si + 1:si + 1 + max_len].reshape(1, -1, act_dim))
                next_r.append(traj['rewards'][si + 1:si + 1 + max_len].reshape(1, -1, 1))

            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))

            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # 填充截断
            n_timesteps.append(np.arange(si + 1, si + 1 + s[-1].shape[1]).reshape(1, -1))
            n_timesteps[-1][n_timesteps[-1] >= max_ep_len] = max_ep_len - 1  # 填充截断

            # 填充和状态归一化
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            next_s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), next_s[-1]], axis=1)
            next_s[-1] = (next_s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            dd[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), dd[-1]], axis=1)
            next_a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., next_a[-1]], axis=1)
            next_r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), next_r[-1]], axis=1)
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            n_timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), n_timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = np.concatenate(s, axis=0)
        next_s = np.concatenate(next_s, axis=0)
        a = np.concatenate(a, axis=0)
        r = np.concatenate(r, axis=0)
        dd = np.concatenate(dd, axis=0)
        next_a = np.concatenate(next_a, axis=0)
        next_r = np.concatenate(next_r, axis=0)
        timesteps = np.concatenate(timesteps, axis=0)
        n_timesteps = np.concatenate(n_timesteps, axis=0)
        mask = np.concatenate(mask, axis=0)

        # 转换为TensorFlow张量
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        next_s = tf.convert_to_tensor(next_s, dtype=tf.float32)
        a = tf.convert_to_tensor(a, dtype=tf.float32) 
        r = tf.convert_to_tensor(r, dtype=tf.float32)
        dd = tf.convert_to_tensor(dd, dtype=tf.float32)
        next_a = tf.convert_to_tensor(next_a, dtype=tf.float32)
        next_r = tf.convert_to_tensor(next_r, dtype=tf.float32)
        timesteps = tf.convert_to_tensor(timesteps, dtype=tf.int32)
        n_timesteps = tf.convert_to_tensor(n_timesteps, dtype=tf.int32)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)

        return s, a, r, dd, next_s, next_a, next_r, timesteps, n_timesteps, mask

    # 创建优化器
    warmup_steps = variant['warmup_steps']
    learning_rate = keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=0.0,
        decay_steps=warmup_steps,
        end_learning_rate=variant['learning_rate'],
    )
    
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        weight_decay=variant['weight_decay']
    )
    
    critic_optimizer = keras.optimizers.Adam(
        learning_rate=variant['critic_learning_rate']
    )

    trainer = KerasSequenceTrainer(
        model=model,
        optimizer=optimizer,
        critic_optimizer=critic_optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        action_dim=act_dim,
        state_dim=state_dim,
        state_mean=state_mean,
        state_std=state_std,
        alpha=variant['alpha'],
    )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='tac-keras',
            config=variant
        )

    for iter in range(variant['max_iters']):
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter + 1, print_logs=True)
        if log_to_wandb:
            wandb.log(outputs)

    # 保存模型
    model.save_weights(group_name+'.h5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kdd')  # kdd, hightech, dow, ndx, mdax, csi
    parser.add_argument('--env', type=str, default='stock')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--u', type=int, default=20) # 40 (kdd, hightech, dow), 20 (ndx, mdax, csi)
    parser.add_argument('--alpha', type=int, default=0.9) # 1.6 (kdd), 2. (hightech), 1.4 (dow), 0.9 (ndx, mdax, csi)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=5)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--critic_learning_rate', type=float, default=1e-6) # 1e-4 (hightech), 1e-6 (others)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=4000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)

    args = parser.parse_args()
    main(vars(args)) 