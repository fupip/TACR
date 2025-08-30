import numpy as np
import torch
from itertools import count

def eval_test(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=512,
        state_mean=0.,
        state_std=1.,
        device='cuda',
    ):

    model.eval()
    model.to(device=device)

    # state_mean = torch.from_numpy(state_mean).to(device=device)
    # state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    state = np.array(state)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    episode_return, episode_length = 0, 0
    total_trade_count = 0
    with torch.no_grad():
        # 抑制过度交易的三个机制参数
        p_thresh = 0.8          # 置信度阈值：最高概率低于该阈值时强制 HOLD
        cooldown_days = 5         # 冷却天数：换仓后至少 N 天内不允许再次换仓
        margin = 0.10             # 滞后区间：买/卖概率需领先其他至少 margin 才允许切换

        # 状态：初始化为 HOLD、不曾交易
        last_trade_day = -10
        last_action_idx = 1       # 0:SELL, 1:HOLD, 2:BUY

        for t in range(max_ep_len):

            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            # print("states",states.shape)
            # print("states nnn",states)
            action = model.get_action(
                states.to(dtype=torch.float32),
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            # print("actions[-1]",actions[-1])
            # print("action",action,type(action))
            # temp_action = action.argmax()
            # print("action argmax",temp_action,type(temp_action))

            actions[-1] = action
            action = action.detach().cpu().numpy()

            # --- 置信度阈值 + 冷却 + 滞后区间 ---
            # 将模型输出转换为概率分布（兼容logits与probabilities）
            vec = np.asarray(action, dtype=np.float64)
            if not (vec >= 0).all() or abs(vec.sum() - 1.0) > 1e-3:
                v = vec - np.max(vec)
                e = np.exp(v)
                probs = e / e.sum()
            else:
                probs = vec

            # 索引约定：0=SELL, 1=HOLD, 2=BUY（与环境一致）
            sell, hold, buy = probs[0], probs[1], probs[2]
            best_idx = int(np.argmax(probs))
            max_prob = float(probs[best_idx])

            # 1) 置信度阈值：低置信度则强制 HOLD
            if max_prob < p_thresh:
                proposed_idx = 1
            else:
                # 2) 滞后区间：只有显著领先才允许买/卖；否则 HOLD
                want_buy = (buy - max(hold, sell)) > margin
                want_sell = (sell - max(hold, buy)) > margin
                if want_buy:
                    proposed_idx = 2
                elif want_sell:
                    proposed_idx = 0
                else:
                    proposed_idx = 1

            # 3) 冷却：短期内禁止再次换仓
            if proposed_idx != last_action_idx and (t - last_trade_day) < cooldown_days:
                final_idx = last_action_idx
            else:
                final_idx = proposed_idx
                if final_idx != last_action_idx:
                    last_trade_day = t

            # 生成最终一键式动作（one-hot）并替换
            final_action = np.zeros_like(probs)
            final_action[final_idx] = 1.0
            action = final_action
            last_action_idx = final_idx

            print("action(after constraints)", action, type(action))

            state, reward, done, result = env.step(action)
            state = np.array(state)
            trade_count = result.get("trade_count", 0)
            if trade_count > 0:
                total_trade_count += 1

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward

            timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

            episode_return += reward
            episode_length += 1
            
            # print(f"episode_return: {episode_return}, episode_length: {episode_length}")

            if done:
                break

    return episode_return, episode_length,total_trade_count