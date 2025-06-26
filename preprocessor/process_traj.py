import numpy as np
from .strategies import create_strategy, MovingAverageStrategy


class trajectory:

    def __init__(
            self,
            dataset,
            df,
            stock_dim,
            state_space,
            action_space,
            tech_indicator_list,
            day=0,
            strategy_name='moving_average',
            strategy_kwargs=None,
    ):

        self.dataset = dataset
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        
        # 初始化策略
        if strategy_kwargs is None:
            strategy_kwargs = {}
        self.strategy_name = strategy_name
        self.strategy_kwargs = strategy_kwargs
        self.strategy = None  # 将在step方法中根据i参数创建

        self.data = self.df.loc[self.day, :]
        self.state = [
                self.data.open,
                self.data.high,
                self.data.low,
                self.data.close,
        ] + [
            self.data[tech]
            for tech in self.tech_indicator_list
        ]
        self.terminal = False
        self.last_day_memory = self.data

    def step(self, i):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)
        if self.terminal:
            return self.state, self.reward, self.terminal, np.array([0.0, 1.0, 0.0])

        else:
            
            self.data = self.df.loc[self.day, :]
            
            self.state = [
                self.data.open,
                self.data.high,
                self.data.low,
                self.data.close,
                ] + [
                    self.data[tech]
                    for tech in self.tech_indicator_list
                ]
            
            # print(self.state)
            # self.terminal = True

            # portion = (self.data.close.values / self.last_day_memory.close.values)
            bc = []

            # i 是生成轨迹的种类而不是step计数
            # i 越大生成的比例越极端
            # 因为 i 从0 开始所以必须+1
            
            # for j in portion:
            #     bc.append(np.exp(j * (i + 1)))   

            # weights = self.softmax_normalization(bc)
            # weights[np.isnan(weights)] = 1.
            
            # ----------- 使用策略系统生成交易信号 -----------
            
            # 根据策略强度参数i创建策略实例（如果还没有创建或参数改变）
            if self.strategy is None or self.strategy.strategy_id != i:
                self.strategy = create_strategy(self.strategy_name, strategy_id=i, **self.strategy_kwargs)
            
            # 使用策略计算持仓和动作
            pos, action = self.strategy.calculate_position_and_action(
                data=self.data, 
                last_day_data=self.last_day_memory
            )

            # 生成完state与weights后向前推进一天
            self.last_day_memory = self.data
            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :] # 获取当天数据,而不是当天之后所有数据

            portfolio_return = ((self.data.close / self.last_day_memory.close) - 1) * pos
            
            # print("pos: ", pos, "portfolio_return: ", portfolio_return)
            
            self.reward = portfolio_return
            
            # print(f"portfolio_return: {portfolio_return}")
        # print("action: ", action)
        return self.state, self.reward, self.terminal, action

    def reset(self):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = [
                self.data.open,
                self.data.high,
                self.data.low,
                self.data.close,
        ] + [
            self.data[tech]
            for tech in self.tech_indicator_list
        ]
        self.terminal = False
        return self.state

    def softmax_normalization(self, actions):
        actions = np.clip(actions, 0, 709)
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output