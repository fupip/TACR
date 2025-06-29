import numpy as np
import wandb
import argparse
import random
import pickle
import pandas as pd
from stock_env.apps import config
from stock_env.allocation.env_portfolio import StockPortfolioEnv
import torch
from tac.models.transformer_actor import TransformerActor
from tac.training.seq_trainer import SequenceTrainer
import torch.backends.cudnn as cudnn

def main(variant):
    mode = variant.get('mode', 'tacr')
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    group_name = f'{env_name}-{dataset}-{mode}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    train = pd.read_csv("datasets/"+dataset+"_train.csv", index_col=[0])
    max_ep_len = train.index[-1]
    
    

    # Load suboptimal trajectories
    dataset_path = f'{"trajectory/" + variant["dataset"] + "_traj.pkl"}'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    state_space=trajectories[0]['observations'].shape[1]  #obs 即 历史state的集合 shape[1]累计的条数,shape[1]指指标的种类
    stock_dimension = len(train.tic.unique())

    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    # Set portfolio allocation environment
    env_kwargs = {
        "dataset": dataset,
        "initial_amount": 1000000,
        "transaction_cost": 0.002,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": 3,
    }
    env = StockPortfolioEnv(df=train, **env_kwargs)

    # Set seed 0
    seed = variant['seed']
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Algorithm 1, line3 : Set the length of the sequence u
    u = variant['u']

    model = TransformerActor(
        state_dim=state_dim,                # 状态空间的维度（输入特征数）  360
        act_dim=act_dim,                    # 动作空间的维度（比如股票数量（每只股票的分配比例）
        max_length=u,                       # 输入序列的最大长度 默认20
        max_ep_len=max_ep_len,              # 一个episode的最大长度（用于时间步embedding等）训练数据条数
        hidden_size=variant['embed_dim'],   # Transformer每层的隐藏单元数（embedding维度）默认128
        n_layer=variant['n_layer'],         # Transformer的层数 默认5
        n_head=variant['n_head'],           # 多头自注意力的头数 默认1
        n_inner=4 * variant['embed_dim'],   # 前馈网络的隐藏层大小（一般是embedding的4倍）
        activation_function=variant['activation_function'], # 激活函数 默认relu
        n_positions=1024,                   # 支持的最大序列长度（位置编码用）
        resid_pdrop=variant['dropout'],     # 残差连接的dropout概率
        attn_pdrop=variant['dropout'],      # 注意力机制的dropout概率
    )

    # 计算每个轨迹的观测值、长度和回报
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    
    # states[0].shape (3733, 360)
    # print("states[0].shape",states[0].shape) 
    

    # used for input normalization
    states = np.concatenate(states, axis=0) # 第1维合并起来,第二维不变 即 (3733*5,360)
    # print("all states.shape",states.shape)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    batch_size = variant['batch_size'] # default 64

    def get_batch(batch_size=64, max_len=u):
        # 随机选择轨迹
        batch_inds = np.random.choice(
            np.arange(len(traj_lens)),
            size=batch_size,
            replace=True
        )
        # print("batch_inds",batch_inds)  # 64个随机数

        s, next_s, next_a, next_r, a, r, d, dd, timesteps, n_timesteps, mask \
            = [], [], [], [], [], [], [], [], [], [], []
            
        for i in range(batch_size):
            traj = trajectories[int(batch_inds[i])] # 5个轨迹中任选一个
            si = random.randint(0, traj['rewards'].shape[0] - 1) # 随机选择一个时间步
            # print("si",si)
            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            
            # print(traj['actions'])
            # print(traj['actions'][si:si + max_len])
            # print(traj['actions'][si:si + max_len].shape)
            # print(act_dim)
            # print(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            
            actions = np.array(traj['actions'][si:si + max_len])
            # if actions.dtype == object:
            #     actions = np.stack(actions, axis=0)
            
            
            a.append(actions.reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            dd.append(traj['dones'][si:si + max_len].reshape(1, -1, 1))
            # print("r",r)
            # print("dd",dd)
            # break

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

            # current_timesteps
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
            
            # next_timesteps
            n_timesteps.append(np.arange(si + 1, si + 1 + s[-1].shape[1]).reshape(1, -1))
            n_timesteps[-1][n_timesteps[-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff

            # padding and state normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            
            # normalize
            s[-1] = (s[-1] - state_mean) / state_std
            next_s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), next_s[-1]], axis=1)
            # normalize
            next_s[-1] = (next_s[-1] - state_mean) / state_std
            
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            dd[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), dd[-1]], axis=1)
            
            next_a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., next_a[-1]], axis=1)
            next_r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), next_r[-1]], axis=1)
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            n_timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), n_timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        next_s = torch.from_numpy(np.concatenate(next_s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        dd = torch.from_numpy(np.concatenate(dd, axis=0)).to(dtype=torch.float32, device=device)
        next_a = torch.from_numpy(np.concatenate(next_a, axis=0)).to(dtype=torch.float32, device=device)
        next_r = torch.from_numpy(np.concatenate(next_r, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        n_timesteps = torch.from_numpy(np.concatenate(n_timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, dd, next_s, next_a, next_r, timesteps, n_timesteps, mask
    # 返回 state ,action reward,dones
    # next_state,next_action,next_reward
    # timesteps,n_timesteps,mask
    
    states,actions,rewards,dones,next_states,next_actions,next_rewards,timesteps,n_timesteps,mask = get_batch(batch_size,u)
    print("*" *50 )
    print("states.shape",states.shape)
    print("actions.shape",actions.shape)
    print("rewards.shape",rewards.shape)
    print("dones.shape",dones.shape)
    print("next_states.shape",next_states.shape)
    print("next_actions.shape",next_actions.shape)
    print("next_rewards.shape",next_rewards.shape)
    print("timesteps.shape",timesteps.shape)
    print("n_timesteps.shape",n_timesteps.shape)
    print("mask.shape",mask.shape)
    print("*" *50 )
    # states.shape torch.Size([64, 20, 360])
    # actions.shape torch.Size([64, 20, 30])
    # rewards.shape torch.Size([64, 20, 1])
    # dones.shape torch.Size([64, 20, 1])
    # next_states.shape torch.Size([64, 20, 360])
    # next_actions.shape torch.Size([64, 20, 30])
    # next_rewards.shape torch.Size([64, 20, 1])
    # timesteps.shape torch.Size([64, 20])
    # n_timesteps.shape torch.Size([64, 20])
    # mask.shape torch.Size([64, 20])
    # return
    

    


    model = model.to(device=device)
    
    # Load pre-trained model if specified
    if variant.get('load_model'):
        model_path = variant['load_model']
        if not model_path.endswith('.pt'):
            model_path += '.pt'
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"load model success: {model_path}")
        except FileNotFoundError:
            print(f"load model failed: {model_path} not found, train from scratch")
        except Exception as e:
            print(f"load model failed: {e}, train from scratch")
    
    warmup_steps = variant['warmup_steps']
    total_steps = variant['max_iters'] * variant['num_steps_per_iter']  # 总训练步数
    
    print(f"Learning Rate Schedule Configuration:")
    print(f"  Initial Learning Rate: {variant['learning_rate']}")
    print(f"  Warmup Steps: {warmup_steps}")
    print(f"  Total Training Steps: {total_steps}")
    print(f"  Decay Steps: {max(0, total_steps - warmup_steps)}")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    
    # Improved learning rate schedule: warmup + cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            # Warmup phase: linear growth
            return (step + 1) / warmup_steps
        elif total_steps <= warmup_steps:
            # If total training steps don't exceed warmup steps, maintain max learning rate
            return 1.0
        else:
            # Post-warmup: cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size=batch_size,
        get_batch=get_batch,
        scheduler=scheduler,
        action_dim=act_dim,
        state_dim=state_dim,
        state_mean=state_mean,
        state_std=state_std,
        alpha=variant['alpha'],
        crtic_lr=variant['critic_learning_rate'],
        mode=mode
    )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='tac',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    for iter in range(variant['max_iters']):
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Iteration {iter + 1}: Learning Rate = {current_lr:.8f}')
        
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter + 1, print_logs=True)
        
        # Add learning rate info to outputs
        outputs['learning_rate'] = current_lr
        
        if log_to_wandb:
            wandb.log(outputs)

    torch.save(trainer.actor.state_dict(), group_name+'.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='csi')  # kdd, hightech, dow, ndx, mdax, csi
    parser.add_argument('--env', type=str, default='stock')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--u', type=int, default=60) # 40 (kdd, hightech, dow), 20 (ndx, mdax, csi)
    parser.add_argument('--alpha', type=float, default=0.9) # 1.6 (kdd), 2. (hightech), 1.4 (dow), 0.9 (ndx, mdax, csi)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=5)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--critic_learning_rate', type=float, default=1e-6) # 1e-4 (hightech), 1e-6 (others)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='tacr')
    parser.add_argument('--load_model', type=str, default=None, help='load model from path')

    args = parser.parse_args()
    print(args)
    main(vars(args))