import json
import typing as t
import pandas as pd
import yfinance as y
import matplotlib.pyplot as plt


def calc_technical_indicators(dataset: t.Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    if type(dataset) == pd.DataFrame:
        series = dataset.iloc[:, 0]
        df = series.to_frame()
        if len(dataset.columns) > 1:
            print("Info: dataset has more that 1 column. Using first column as a source for calculations")
    elif type(dataset) == pd.Series:
        series = dataset
        df = series.to_frame()
        # assert type(series) == pd.Series, "series should be of type pd.Series"
    else:
        raise Exception('Wrong dataset type: {}'.format(type(dataset)))

    # Create 7 and 21 days Moving Average
    df['ma7'] = series.rolling(window=7).mean()
    df['ma21'] = series.rolling(window=21).mean()

    # Create MACD
    df['26ema'] = series.ewm(span=26).mean()
    df['12ema'] = series.ewm(span=12).mean()
    df['MACD'] = df['12ema'] - df['26ema']

    # Create Bollinger Bands
    df['20sd'] = series.rolling(window=20).std()
    df['upper_band'] = df['ma21'] + (df['20sd'] * 2)
    df['lower_band'] = df['ma21'] - (df['20sd'] * 2)

    # Create Exponential moving average
    df['ema'] = series.ewm(com=0.5).mean()

    # Create Momentum
    # df['momentum'] = series - 1

    return df.drop(df.columns[0], axis=1)  # remove initial column


def plot_technical_indicators(dataset):
    plt.figure(figsize=(16, 10), dpi=100)
    shape_0 = dataset.shape[0]
    xmacd_ = shape_0

    x_ = range(3, dataset.shape[0])
    x_ =list(dataset.index)

    # Plot first subplot
    plt.subplot(2, 1, 1)
    plt.plot(dataset['ma7'], label='MA 7', color='g',linestyle='--')
    plt.plot(dataset['price'], label='Closing Price', color='b')
    plt.plot(dataset['ma21'], label='MA 21', color='r', linestyle='--')
    plt.plot(dataset['upper_band'], label='Upper Band', color='c')
    plt.plot(dataset['lower_band'], label='Lower Band', color='c')
    plt.fill_between(x_, dataset['lower_band'], dataset['upper_band'], alpha=0.35)
    plt.title('Technical indicators for Goldman Sachs.')
    plt.ylabel('USD')
    plt.legend()

    # Plot second subplot
    plt.subplot(2, 1, 2)
    plt.title('MACD')
    plt.plot(dataset['MACD'], label='MACD', linestyle='-.')
    # plt.hlines(15, xmacd_, shape_0, colors='g', linestyles='--')
    # plt.hlines(-15, xmacd_, shape_0, colors='g', linestyles='--')
    # plt.plot(dataset['momentum'], label='Momentum', color='b', linestyle='-')

    plt.legend()
    plt.show()


if __name__ == "__main__":

    plt.style.use("dark_background")

    df = y.Ticker('UPWK').history(period='max')[['Close']].rename(columns={'Close': 'price'})

    technicals = calc_technical_indicators(df)

    plot_technical_indicators(pd.concat([df, technicals], axis=1))
