import numpy as np
from collections import deque
from .base_strategy import BaseStrategy


class HuricStrategy(BaseStrategy):
    """
    Huric Strategy (Daily, Close-based breakout)
    固定使用 MA20 与 MA60
    - 用 MA20 与 MA60 交叉 + 平滑产生方向（BuySetup/SellSetup）
    - 以最近20日的高点/低点 ± pnt% 作为突破阈值
    - 收盘价突破阈值才入场，成交价按收盘
    """

    def __init__(
        self,
        strategy_id=0,
        pnt=0.5,                # 默认 0.5%，可调
        smooth_n=5,             # 对 MA 再平滑的窗口
        reverse_on_opposite=True,
        skipstep_on_first=True,
        actionqueue_len=2
    ):
        super().__init__(strategy_id=strategy_id)  # 保持接口一致
        self.pnt = float(pnt)
        self.smooth_n = int(smooth_n)
        self.reverse_on_opposite = bool(reverse_on_opposite)
        self.skipstep_on_first = bool(skipstep_on_first)

        # 状态
        self.pos_state = 0.0
        self.hold_time = 0
        self.actionqueue = deque([0], maxlen=max(actionqueue_len, 1))

        self.buy_setup = False
        self.sell_setup = False
        self.LEPrice = 0.0
        self.SEPrice = 0.0
        self._skip_buy_once = False
        self._skip_sell_once = False

        # 平滑器缓存
        self._buf_fast = deque(maxlen=self.smooth_n)
        self._buf_slow = deque(maxlen=self.smooth_n)
        self._prev_MAMA = None
        self._prev_MAMA2 = None

        # 最近N日高低点
        self._roll_window = 5
        self._recent_high = deque()
        self._recent_low = deque()
        self._roll_high = None
        self._roll_low = None

    # 工具
    @staticmethod
    def _avg(buf):
        return sum(buf) / len(buf) if buf else None

    @staticmethod
    def _pct(v, pct):
        return v * (1.0 + pct / 100.0)

    def _smooth_push(self, fast_ma, slow_ma):
        self._buf_fast.append(fast_ma)
        self._buf_slow.append(slow_ma)
        return self._avg(self._buf_fast), self._avg(self._buf_slow)

    def _detect_cross(self, prev_a, prev_b, now_a, now_b):
        if None in (prev_a, prev_b, now_a, now_b):
            return False, False
        up = (prev_a <= prev_b) and (now_a > now_b)
        down = (prev_a >= prev_b) and (now_a < now_b)
        return up, down

    def _push_hl(self, high_val, low_val):
        self._recent_high.append(high_val)
        self._recent_low.append(low_val)
        if len(self._recent_high) > self._roll_window:
            self._recent_high.popleft()
            self._recent_low.popleft()
        self._roll_high = max(self._recent_high)
        self._roll_low = min(self._recent_low)

    def calculate_position_and_action(self, data, last_day_data=None):
        """
        data 需包含:
        - 'close','high','low'
        - 'close_20_sma','close_60_sma'
        """
        close_ = float(data['close'])
        high_ = float(data['high'])
        low_ = float(data['low'])
        ma20 = float(data['close_20_sma'])
        ma60 = float(data['close_60_sma'])

        # 更新滚动高低点
        self._push_hl(high_, low_)

        # 计算平滑 MA
        MAMA_now, MAMA2_now = self._smooth_push(ma20, ma60)
        up_cross, down_cross = self._detect_cross(self._prev_MAMA, self._prev_MAMA2, MAMA_now, MAMA2_now)

        # 交叉产生 Setup 和阈值
        if up_cross and self._roll_high is not None:
            self.buy_setup = True
            self.sell_setup = False
            self.LEPrice = self._pct(self._roll_high, +self.pnt)
            self.SEPrice = 0.0
            if self.skipstep_on_first:
                self._skip_buy_once = True

        if down_cross and self._roll_low is not None:
            self.sell_setup = True
            self.buy_setup = False
            self.SEPrice = self._pct(self._roll_low, -self.pnt)
            self.LEPrice = 0.0
            if self.skipstep_on_first:
                self._skip_sell_once = True

        preaction = self.actionqueue[0]
        action_vec = None

        # 入场：收盘突破
        if self.pos_state == 0:
            if self.buy_setup and close_ >= self.LEPrice and preaction != +1:
                if self._skip_buy_once:
                    self._skip_buy_once = False
                else:
                    self.pos_state = +1
                    action_vec = np.array([0.0, 0.0, 1.0])
                    self.actionqueue.append(+1)

            if (action_vec is None) and self.sell_setup and close_ <= self.SEPrice and preaction != -1:
                if self._skip_sell_once:
                    self._skip_sell_once = False
                else:
                    self.pos_state = -1
                    action_vec = np.array([1.0, 0.0, 0.0])
                    self.actionqueue.append(-1)

        elif self.pos_state < 0:
            if self.buy_setup and close_ >= self.LEPrice:
                if self.reverse_on_opposite:
                    self.pos_state = +1
                    action_vec = np.array([0.0, 0.0, 1.0])
                    self.actionqueue.append(+1)
                else:
                    self.pos_state = 0
                    action_vec = np.array([0.0, 1.0, 0.0])
                    self.actionqueue.append(0)

        elif self.pos_state > 0:
            if self.sell_setup and close_ <= self.SEPrice:
                if self.reverse_on_opposite:
                    self.pos_state = -1
                    action_vec = np.array([1.0, 0.0, 0.0])
                    self.actionqueue.append(-1)
                else:
                    self.pos_state = 0
                    action_vec = np.array([0.0, 1.0, 0.0])
                    self.actionqueue.append(0)

        # 未触发则保持
        if action_vec is None:
            action_vec = np.array([0.0, 1.0, 0.0])

        self.hold_time = 0 if self.pos_state == 0 else (self.hold_time + 1)
        self._prev_MAMA, self._prev_MAMA2 = MAMA_now, MAMA2_now

        return self.pos_state, action_vec

    def get_strategy_info(self):
        return {
            'description': 'Huric Strategy (MA20/MA60, close-based breakout)',
            'pnt_percent': self.pnt,
            'smooth_n': self.smooth_n,
            'reverse_on_opposite': self.reverse_on_opposite,
            'skipstep_on_first': self.skipstep_on_first
        }
