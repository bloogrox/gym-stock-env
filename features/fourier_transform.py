import numpy as np
import pandas as pd


def inverse_fourier_transform_for(data, componens_num: int) -> np.array:
    if type(data) not in [pd.Series, np.array, list]:
        raise Exception('Unacceptable `data` type')

    def zero_fill(ndarr, num):
        a = np.copy(ndarr)
        a[num:-num] = 0
        return a
    fft = np.fft.fft(data)
    ifft = np.fft.ifft(zero_fill(fft, componens_num))
    return ifft
