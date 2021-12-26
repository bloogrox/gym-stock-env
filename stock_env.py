import gym
from gym import spaces
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt


def _pct_change(sale_price: float, basis_price: float) -> float:
    ret = (sale_price - basis_price) / basis_price
    return ret

# def _buy_and_hold_returns(sale_price: float, basis_price: float) -> float:
#     # basis_price = df['price'][0]
#     # sale_price = df['price'][-1]
#     ret = (sale_price - basis_price) / basis_price
#     return ret


# def buy_and_hold_returns(df: pd.DataFrame, timestep: int) -> float:
#     basis_price = df['price'][0]
#     sale_price = df['price'][timestep - 1]
#     ret = _pct_change(sale_price, basis_price)
#     # ret = (sale_price - basis_price) / basis_price
#     return ret


# def excess_return(
#         net_worth: float,
#         initial_balance: float,
#         df: pd.DataFrame,
#         timestep: int
#     ) -> float:
#     """
#     Rbh - buy-and-hold returns
#     """
#     R = _pct_change(net_worth, initial_balance)
#     Rbh = buy_and_hold_returns(df, timestep)
#     return R - Rbh


def _vectorize(state: dict) -> np.array:
    return np.concatenate((
        np.array([
            state['balance'],
            state['shares'],
            state['price']
        ]),
        state['features'],
    ))


# def _devectorize(arr: np.array) -> dict:
#     return {
#         'balance': arr[0],
#         'shares': arr[1],
#         'price': arr[2],
#         'price': arr[2],
#     }

def _net_worth(state: dict) -> float:
    return state['balance'] + state['price'] * state['shares']


class StockEnv(gym.Env):

    def __init__(self, df, initial_balance=0):
        # action: [trading_decision, percentage]
        # percentage:
        #   [-1; 1] => [0; 1]
        #   percentage of balance to buy stock
        #   or percentage of shares to sell
        self.action_space = spaces.Box(low=np.array([-1, -1], dtype=np.float64), high=np.array([1, 1], dtype=np.float64), dtype=np.float64)
        box_size = len(df.columns) + 2  # balance, shares, all features from dataframe
        self.observation_space = spaces.Box(0, np.inf, shape=(box_size,), dtype=np.float32)

        self.initial_balance = initial_balance

        self.df = df
        self.timestep = 0
        self.state = {}
        self.trades = []

        self.reset()

    def reset(self):
        self.timestep = 0
        self.state = {
            'balance': self.initial_balance,
            'shares': 0,
            'price': self.df.iloc[0]['price'],
            'features': self.df.iloc[0][1:].values,
        }
        self.trades = []
        return _vectorize(self.state)

    def step(self, action):
        trading_decision_scaled = action[0]
        percentage_scaled = action[1]

        self.timestep += 1
        # todo depends on df format
        self.state['price'] = self.df.iloc[self.timestep-1]['price']
        self.state['features'] = self.df.iloc[self.timestep-1][1:].values
        done = self.timestep >= len(self.df.index)

        trading_decision = np.digitize(trading_decision_scaled, np.linspace(-1, 1, 4))
        percentage = np.interp(percentage_scaled, (-1, 1), (0, 1))

        if trading_decision == 1:  # buy
            amount = self.state['balance'] * percentage
            shares_count = amount // self.state['price']
            if shares_count:
                shares_price = shares_count * self.state['price']
                self.state['balance'] -= shares_price
                self.state['shares'] += shares_count
                trade = {
                    'timestep': self.timestep,
                    'type': 'buy',
                    'shares': shares_count,
                    'price': self.state['price'],
                    'amount': shares_price,
                }
                self.trades.append(trade)
        elif trading_decision == 2:  # sell
            shares_count = round(self.state['shares'] * percentage)
            if shares_count:
                shares_price = shares_count * self.state['price']
                self.state['balance'] += shares_price
                self.state['shares'] -= shares_count
                trade = {
                    'timestep': self.timestep,
                    'type': 'sell',
                    'shares': shares_count,
                    'price': self.state['price'],
                    'amount': shares_price,
                }
                self.trades.append(trade)
        else:  # hold
            pass

        obs = _vectorize(self.state)

        """Reward
        https://ai.stackexchange.com/a/10912
        """
        # reward = _net_worth(self.state) - self.initial_balance
        reward = self._excess_return()

        info = {
            'timestep': self.timestep,
            'timeseries length': len(self.df.index),
            'action': {
                'raw': action,
                'trading_decision': trading_decision,
                'percentage': percentage,
            },
            'reward': reward,
            'NetWorth': _net_worth(self.state),
        }
        return obs, reward, done, info

    def render(self):
        print(f"##### Timestep {self.timestep}")
        print(f"Trading period: {self.df.index[0].date()} – {self.df.index[self.timestep - 1].date()}")
        # print(tabulate(self.trades, headers="keys"))
        print(tabulate([self.state], headers="keys"))
        print(f"Net Worth: {self._net_worth():.2f}")
        print(f"Profit: {100 * self._portfolio_return():.1f}%")
        print(f"Buy-and-Hold: {100 * self._buy_and_hold_return():.1f}%")
        print(f"Excess return: {100 * self._excess_return():.1f}%")
        print("Total trades: ", len(self.trades))
        ax = self.df[['price']].iloc[:self.timestep].plot()
        df_trades = pd.DataFrame(columns=['buy', 'sell'])
        for trade in self.trades:
            series = self.df.iloc[int(trade['timestep'])-1]
            ind = series.name
            # print(type(ind))
            val = series['price']
            column = trade['type']
            df_trades.loc[ind, column] = val
        df_trades.astype('float').plot(ax=ax, style='.', color={'buy': 'limegreen', 'sell': 'red'})
        plt.show()


    def _net_worth(self):
        return self.state['balance'] + self.state['price'] * self.state['shares']

    def _buy_and_hold_return(self):
        basis_price = self.df['price'][0]
        sale_price = self.df['price'][self.timestep - 1]
        return _pct_change(sale_price, basis_price)

    def _portfolio_return(self):
        return _pct_change(self._net_worth(), self.initial_balance)

    def _excess_return(self):
        R = self._portfolio_return()  # _pct_change(self._net_worth(), self.initial_balance)
        Rbh = self._buy_and_hold_return()
        return R - Rbh


if __name__ == '__main__':
    import pandas as pd
    from stable_baselines3.common.env_checker import check_env
    env = StockEnv(pd.DataFrame({'price': [1, 1]}))
    check_env(env, warn=True)
