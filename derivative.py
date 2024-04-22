import matplotlib.pyplot as plt
import pandas as pd
import util
import numpy as np
from math import e, log
from scipy.signal import spectrogram

def calculate_pump_performance(pressure_start, pressure_end, pressure_start2, pressure_end2, pressure_start3, pressure_end3):
    return (pressure_start - pressure_end) / (pressure_start2 - pressure_end2)

df_05Hz = pd.read_csv('data/new/4,4-4,0-0,5G.csv', index_col=None, header=None)
df_1Hz = pd.read_csv('data/new/4,4-4,35-1G.csv', index_col=None, header=None)
df_2Hz = pd.read_csv('data/new/4,4-4,35-2G.csv', index_col=None, header=None)

v_05Hz = df_05Hz.to_numpy().flatten()
v_1Hz = df_1Hz.to_numpy().flatten()
v_2Hz = df_2Hz.to_numpy().flatten()

def get_most_frequent(arr: np.ndarray, percent: int) -> np.ndarray:
    nthtile = np.percentile(arr, q=[percent, 100 - percent])
    return arr[(arr > nthtile[0]) & (arr < nthtile[1])]

def get_n_mid_pulses(pressure: np.ndarray, n: int = 1, pulse_cnt: int = 1000) -> np.ndarray:
    pulse_len = n * len(pressure) // pulse_cnt
    return pressure[len(pressure) // 2 - pulse_len // 2 : len(pressure) // 2 + pulse_len // 2]

def get_smoothed_pulses(pressure: np.ndarray, window_size: int) -> np.ndarray:
    return np.convolve(pressure, np.ones(window_size) / window_size, mode='valid')

def process_pressure_mid_pulses(pressure_data: np.ndarray, impulses_cnt: int = 2, window_size: int = 5): 
    imp_mid = get_n_mid_pulses(pressure_data, impulses_cnt)
    imp_mid_smoothed = get_smoothed_pulses(imp_mid, window_size)
    imp_mid_smoothed_dbl = get_smoothed_pulses(imp_mid_smoothed, window_size)

    imp_mid_grad = np.gradient(imp_mid)
    imp_mid_smoothed_grad = np.gradient(imp_mid_smoothed)
    imp_mid_smoothed_dbl_grad = np.gradient(imp_mid_smoothed_dbl)

    return [imp_mid, imp_mid_smoothed, imp_mid_smoothed_dbl, imp_mid_grad, imp_mid_smoothed_grad, imp_mid_smoothed_dbl_grad]

pulse_05Hz = process_pressure_mid_pulses(v_05Hz, 2)
pulse_1Hz = process_pressure_mid_pulses(v_1Hz, 4)
pulse_2Hz = process_pressure_mid_pulses(v_2Hz, 8)

plt.subplot(3, 1, 1)
plt.plot(pulse_05Hz[2])
plt.title('Pressure 0.5 Hz')
plt.subplot(3, 1, 2)
plt.plot(pulse_1Hz[2])
plt.title('Pressure 1 Hz')
plt.subplot(3, 1, 3)
plt.plot(pulse_2Hz[2])
plt.title('Pressure 2 Hz')
plt.show()

plt.subplot(3, 1, 1)
plt.plot(pulse_05Hz[5])
plt.title('Pressure gradient 0.5 Hz')
plt.subplot(3, 1, 2)
plt.plot(pulse_1Hz[5])
plt.title('Pressure gradient 1 Hz')
plt.subplot(3, 1, 3)
plt.plot(pulse_2Hz[5])
plt.title('Pressure gradient 2 Hz')
plt.show()