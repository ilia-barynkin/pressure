import pandas as pd
from scipy import stats, signal
import numpy as np
import matplotlib.pyplot as plt

def remove_outliers(df):
    df = df.drop_duplicates()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col][(np.abs(stats.zscore(df[col])) < 3)]
    return df

def interpolate_outliers(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_zscore = np.abs(stats.zscore(df[col]))
            outliers = col_zscore > 3
            df[col] = df[col].mask(outliers).interpolate()
    return df

def butter_highpass(data:np.ndarray, cutoff=1, fs=200, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, data)
    
def moving_average_filter(df, sampling_rate=200, window_size=5):
    # Coefficients for the moving average filter
    b = np.ones(window_size) / window_size
    a = 1
    
    filtered_df = pd.DataFrame(index=df.index)
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            zi = lfilter_zi(b, a) * df[column].iloc[0]
            filtered_column, _ = lfilter(b, a, df[column], zi=zi)
            filtered_df[column] = filtered_column
    
    return filtered_df

def calc_running_integral(values: np.ndarray) -> np.ndarray:
    return np.cumsum(values)

def plot_2_ndarrays(y1 : np.ndarray, y2 : np.ndarray):
    plt.subplot(3, 1, 1)
    plt.plot(y1)
    plt.subplot(3, 1, 2)
    plt.plot(y2)
    plt.subplot(3, 1, 3)
    plt.plot(y1)
    plt.plot(y2)
    plt.show()

def plot_dependency(x: np.ndarray, y: np.ndarray):
    plt.plot(x, y)
    plt.show()

def add_missing_rows(df1, df2):
    additional_rows = len(df1) - len(df2)
    if additional_rows > 0:
        zeros_df = pd.DataFrame(np.zeros((additional_rows, df2.shape[1])), columns=df2.columns)
        df2 = pd.concat([df2, zeros_df], ignore_index=True)
    return (df1, df2)

