import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import e, log, tan
import os
import re

def get_smoothed_pulses(pressure: np.ndarray, window_size: int) -> np.ndarray:
    return np.convolve(pressure, window_size, mode='valid')

#vals_stable = vals[9500:10500]
#vals_smooth = get_smoothed_pulses(vals_stable, 5)

#print(vals_stable.max() - vals_stable.min())

folder_path = 'data/logs'
file_list = os.listdir(folder_path)

#df = pd.DataFrame(['pump_name', 'frequency', 'values', 'smoothed', 'stable'])

i = 0

rows = []

def parse_file(filename):
    parts = filename.split('_')
    pump_name = parts[10]
    frequency = parts[-2]
    frequency_f = float(frequency.replace(',', '.').replace('г', '').replace('Г', ''))
    values = pd.read_csv('data/logs/' + filename, delimiter=';', header=None, index_col=None, decimal=',').to_numpy().flatten()
    smoothed = get_smoothed_pulses(values, 10)
    stable = values[(len(smoothed) // 2) - 100: (len(smoothed) // 2) + 100]
    amp = stable.max() - stable.min()
    amp_whole = values.max() - values.min()
    mean = stable.mean()
    std_div = np.std(values)
    return [pump_name,frequency_f, values, smoothed, stable, amp, amp_whole, mean, std_div]

for name in file_list:
    row = parse_file(name)
    rows.append(row)

df = pd.DataFrame.from_records(rows)
df.columns = ['pump_name','frequency', 'values', 'smoothed', 'stable', 'amp', 'amp_whole', 'mean', 'std_div']

import matplotlib.pyplot as plt

# Plot 'values' column for each row
grouped = df.groupby('pump_name')

# for name, group in grouped:
#     fig, ax1 = plt.subplots()

#     color = 'tab:red'
#     ax1.set_xlabel('frequency (Hz)')
#     ax1.set_ylabel('mean', color=color)
#     ax1.plot(group['frequency'], group['mean'], 'o-', color=color)
#     ax1.tick_params(axis='y', labelcolor=color)

#     ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#     color = 'tab:blue'
#     ax2.set_ylabel('amp', color=color)  # we already handled the x-label with ax1
#     ax2.plot(group['frequency'], group['amp'], 's-', color=color)
#     ax2.tick_params(axis='y', labelcolor=color)

#     ax3 = ax1.twinx()  # instantiate a third axes that shares the same x-axis
#     ax3.spines['right'].set_position(('outward', 60))  # Offset the right spine of ax3
#     color = 'tab:green'
#     ax3.set_ylabel('std_div', color=color)
#     ax3.plot(group['frequency'], group['std_div'], '^-', color=color)
#     ax3.tick_params(axis='y', labelcolor=color)

#     fig.tight_layout()  # to make sure that the labels do not overlap
#     plt.title(f'Pump {name} - Mean, Amp, Std_div vs Frequency')
#     plt.show()

# Plot 'mean' vs 'frequency' for each pump in 'grouped' and label each line with the pump names
fig, ax = plt.subplots()

for name, group in grouped:
    ax.plot(group['frequency'], group['mean'], label=name)

ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Mean')
ax.set_title('Mean vs Frequency for Each Pump')
ax.legend()
plt.show()
