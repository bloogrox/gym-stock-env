import json
import numpy as np
import pandas as pd
import yfinance as y
from pprint import pprint
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv
from stock_env import StockEnv
from features.arima import arima_predict
from features.technical import calc_technical_indicators
from features.fourier_transform import inverse_fourier_transform_for


def train_test_split(df: pd.DataFrame, train_size=None):
    split_position = int(len(df.index) * train_size)
    train_df = df.iloc[:split_position]
    test_df = df.iloc[split_position:]
    return train_df, test_df


def load_ticker(ticker: str, start=None) -> pd.DataFrame:
    df = (
        y.Ticker(ticker)
        .history(start=start)[['Close']]
        .rename(columns={'Close': ticker})
    )
    return df


if __name__ == '__main__':

    plt.style.use("dark_background")

    start = '2017-01-01'

    df = (
        y.Ticker('BABA')
        .history(start=start)[['Close']]
        .rename(columns={'Close': 'price'})
    )


    # arima predictions
    print("start arima predictions")
    train_arima_size = 0.2
    min_periods = int(len(df.index) * train_arima_size)
    df['arima'] = df['price'].expanding(min_periods=min_periods).apply(arima_predict)
    print("arima predictions finished")

    # Fourier transforms
    df = (df
        .assign(ifft_3=np.real(inverse_fourier_transform_for(df['price'], 3)))
        .assign(ifft_6=np.real(inverse_fourier_transform_for(df['price'], 6)))
        .assign(ifft_9=np.real(inverse_fourier_transform_for(df['price'], 9)))
    )

    # technical indicators
    technicals = calc_technical_indicators(df['price'])

    # correlated assets
    print("load indexes, indicators, currencies, other assets")

    snp500 = load_ticker('^GSPC', start)
    nyse = load_ticker('^NYA', start)
    nasdaq = load_ticker('^IXIC', start)
    vix = load_ticker('^VIX', start)
    usdeur = load_ticker('EUR=X', start)
    usdgbp = load_ticker('GBP=X', start)


    df = pd.concat([df, snp500, nyse, nasdaq, usdgbp, usdeur, vix, technicals], axis=1)

    # ---
    df = df.dropna()

    train_df, test_df = train_test_split(df, train_size=0.8)


    # train
    print("start training")
    # env = SubprocVecEnv([lambda: StockEnv(train_df, initial_balance=10000)])
    env = StockEnv(train_df, initial_balance=10000)
    hyperparams = {'batch_size': 128, 'n_steps': 1024, 'gamma': 0.9999, 'learning_rate': 0.2206586009898024, 'ent_coef': 3.9795610358689176e-07, 'clip_range': 0.3, 'n_epochs': 5, 'gae_lambda': 0.92, 'max_grad_norm': 0.8, 'vf_coef': 0.9033795646712146}
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/", **hyperparams)
    # model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=30000)  # <<<--- TIMESTEPS !!!
    print("training finished")

    # todo validation
    # todo test on unseen data

    # test
    env = StockEnv(test_df, initial_balance=10000)
    obs = env.reset()
    while True:
        while True:
            action, _ = model.predict(obs)
            obs, r, done, info = env.step(action)
            # env.render()
            # pprint(info)
            # input()
            if done:
                env.render()
                # pprint(info)
                print("finished")
                break
        print("press enter to play again")
        input()
        obs = env.reset()

