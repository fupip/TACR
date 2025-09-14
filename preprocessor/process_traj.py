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
            transaction_cost=0.001,
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
        
        self.transaction_cost = transaction_cost
        
        # 初始化策略
        if strategy_kwargs is None:
            strategy_kwargs = {}
        self.strategy_name = strategy_name
        self.strategy_kwargs = strategy_kwargs
        self.strategy = None  # 将在step方法中根据i参数创建

        self.data = self.df.loc[self.day, :]
        # self.state = [
        #         self.data.open_z,
        #         self.data.high_z,
        #         self.data.low_z,
        #         self.data.close_z,
        # ] + [
        #     self.data[tech]
        #     for tech in self.tech_indicator_list
        # ]
        self.last_pos = 0
        delta_close = self.data.close - self.data.close_60_sma
        self.state = [
            delta_close,self.last_pos
        ]
        
        print("process_traj state: ", self.state)
        self.terminal = False
        self.last_day_memory = self.data
        

    def step(self, i):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print(actions)
        if self.terminal:
            return self.state, self.reward, self.terminal,np.array([0.0, 1.0, 0.0]),0,0,0

        else:
            # print("#### self.day",self.day)
            self.data = self.df.loc[self.day, :]
            self.state = self.data
            # print("#### self.data",self.data)
            # self.state = [
            #     self.data.open_z,
            #     self.data.high_z,
            #     self.data.low_z,
            #     self.data.close_z,
            #     ] + [
            #         self.data[tech]
            #         for tech in self.tech_indicator_list
            #     ]
            
            delta_close = (self.data.close - self.data.close_60_sma)/self.data.close_60_sma
            # self.state = [
            #     delta_close,self.last_pos
            # ]
            
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
            
            trade_flag = abs(pos - self.last_pos)
            trade_count = 0 
            if abs(pos - self.last_pos) > 0:
                trade_count = 1
            self.last_pos = pos

            # 生成完state与weights后向前推进一天
            self.last_day_memory = self.data
            # load next state
            self.day += 1
            self.data = self.df.loc[self.day, :] # 获取当天数据,而不是当天之后所有数据
            trade_fee  = trade_flag * self.transaction_cost
            portfolio_return_nofee = ((self.data.close / self.last_day_memory.close) - 1) * pos
            
            portfolio_return_with_fee = ((self.data.close / self.last_day_memory.close) - 1) * pos - trade_fee
            
            # print("pos: ", pos, "portfolio_return: ", portfolio_return)
            
            self.reward = portfolio_return_with_fee
            
            # print(f"portfolio_return: {portfolio_return}")
        # 在推进到下一天后，重算“下一时刻”的 state 并返回

        delta_close_next = (self.data.close - self.last_day_memory.close_60_sma)/self.last_day_memory.close_60_sma if hasattr(self.last_day_memory, 'close_60_sma') else (self.data.close - self.last_day_memory.close)
        # 更稳妥地用当前 self.data 计算
        delta_close_next = (self.data.close - self.data.close_60_sma)/self.data.close_60_sma
        self.state = [
            delta_close_next, self.last_pos
        ]
        # 更新终止标记到新的一天
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        # print("state: ", self.state)
        return self.state, self.reward, self.terminal, action,pos,trade_count,portfolio_return_nofee


    def reset(self):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        # self.state = [
        #         self.data.open_z,
        #         self.data.high_z,
        #         self.data.low_z,
        #         self.data.close_z,
        # ] + [
        #     self.data[tech]
        #     for tech in self.tech_indicator_list
        # ]
        
        delta_close = (self.data.close - self.data.close_60_sma)/self.data.close_60_sma
        self.state = [
            delta_close,0
        ]
        
        print("process_traj reset state: ", self.state)
        self.terminal = False
        return self.state

    def softmax_normalization(self, actions):
        actions = np.clip(actions, 0, 709)
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output