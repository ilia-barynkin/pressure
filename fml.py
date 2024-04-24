import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import e, log, tan, sqrt
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def get_smoothed_pulses(pressure: np.ndarray, window_size: int) -> np.ndarray:
    return np.convolve(pressure, window_size, mode='valid')

folder_path = 'data/logs'
file_list = os.listdir(folder_path)

i = 0
rows = []

def get_maxima(values: np.ndarray):
    # Find local maxima
    peaks, _ = find_peaks(values, distance = len(values) / 20)
    # Generate array with NaN values
    maxima_line = np.full(values.shape, np.nan)
    # Set the local maxima points
    maxima_line[peaks] = values[peaks]
    # Interpolate to form a continuous line through the local maxima
    maxima_line = np.interp(np.arange(len(values)), peaks, values[peaks])

    return maxima_line

def get_minima(values: np.ndarray):
    # Find local minima
    peaks, _ = find_peaks(-values, distance = len(values) / 20)
    # Generate array with NaN values
    minima_line = np.full(values.shape, np.nan)
    # Set the local minima points
    minima_line[peaks] = values[peaks]
    # Interpolate to form a continuous line through the local minima
    minima_line = np.interp(np.arange(len(values)), peaks, values[peaks])

    return minima_line

def remove_outliers(values: np.ndarray):
    mean = np.mean(values)
    std_dev = np.std(values)
    values = values[(values > mean - 2 * std_dev * 1.3) & (values < mean + 2 * std_dev * 1.3)]
    return values

def parse_file(filename):
    parts = filename.split('_')
    pump_name = parts[7]
    nominal = parts[-3]
    frequency = parts[-2]
    frequency_f = float(frequency.replace(',', '.').replace('г', '').replace('Г', ''))
    values = pd.read_csv('data/logs/' + filename, delimiter=';', header=None, index_col=None, decimal=',').to_numpy().flatten()
    smoothed = get_smoothed_pulses(values, 20)
    smoothed_dbl = get_smoothed_pulses(smoothed, 20)
    stable = values[(len(values) // 2) - int(1000 / frequency_f): (len(values) // 2) + int(1000 / frequency_f)]
    amp = stable.max() - stable.min()
    amp_whole = values.max() - values.min()
    grad_t = np.gradient(values) 
    grad_t_smoothed = np.gradient(smoothed_dbl)
    tail = remove_outliers(values[len(values) * 2 // 3: len(values) * 8 // 9])
    mean = tail.mean()
    maxima_tail = get_maxima(remove_outliers(tail))
    minima_tail = get_minima(remove_outliers(tail))
    mean_maxima_tail = maxima_tail.mean()
    mean_minima_tail = minima_tail.mean()
    
    std_dev = np.std(tail)
    delta_p = 2 * sqrt(std_dev)
    return [pump_name, nominal, frequency_f, values, smoothed, smoothed_dbl, stable, amp, amp_whole, mean, std_dev, grad_t, grad_t_smoothed, tail, delta_p, mean_maxima_tail, mean_minima_tail]

for name in file_list:
    row = parse_file(name)
    rows.append(row)

df = pd.DataFrame.from_records(rows)
df.columns = ['pump_name', 'nominal', 'frequency_f', 'values', 'smoothed', 'smoothed_dbl', 'stable', 'amp', 'amp_whole', 'mean', 'std_dev', 'grad_t', 'grad_t_smoothed', 'tail', 'delta_p', 'mean_maxima_tail', 'mean_minima_tail']
df.head()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_tail_frequencies(df, group_column='pump_name'):
    """Plot mean_tail, mean_minima_tail, and mean_maxima_tail vs frequency for each group."""
    
    # Set up the matplotlib figure
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 15), sharex=True)
    
    # Customize the style
    sns.set_style("whitegrid")
    
    # Plot mean_tail vs frequency
    sns.lineplot(ax=axes[0], data=df, x='frequency_f', y='mean', hue=group_column, palette="viridis")
    axes[0].set_title('Mean Tail vs Frequency')
    axes[0].set_ylabel('Mean Tail')
    
    # Plot mean_minima_tail vs frequency
    sns.lineplot(ax=axes[1], data=df, x='frequency_f', y='mean_minima_tail', hue=group_column, palette="viridis")
    axes[1].set_title('Mean Minima Tail vs Frequency')
    axes[1].set_ylabel('Mean Minima Tail')
    
    # Plot mean_maxima_tail vs frequency
    sns.lineplot(ax=axes[2], data=df, x='frequency_f', y='mean_maxima_tail', hue=group_column, palette="viridis")
    axes[2].set_title('Mean Maxima Tail vs Frequency')
    axes[2].set_xlabel('Frequency (f)')
    axes[2].set_ylabel('Mean Maxima Tail')
    
    # Adjust the layout
    plt.tight_layout()
    plt.show()

# Assuming 'df' is the DataFrame and 'pump_name' is the group column
plot_tail_frequencies(df, 'pump_name')

def plot_aggregated_tails(df, pump_name):
    """Plot mean, mean_maxima_tail, and mean_minima_tail vs frequency for one particular pump."""
    
    # Filter the DataFrame for the specified pump_name
    df_pump = df[df['pump_name'] == pump_name]
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot mean vs frequency
    sns.lineplot(ax=ax, data=df_pump, x='frequency_f', y='mean', label='Mean', palette="viridis")
    
    # Plot mean_minima_tail vs frequency
    sns.lineplot(ax=ax, data=df_pump, x='frequency_f', y='mean_minima_tail', label='Minima Mean', palette="viridis")
    
    # Plot mean_maxima_tail vs frequency
    sns.lineplot(ax=ax, data=df_pump, x='frequency_f', y='mean_maxima_tail', label='Maxima Mean', palette="viridis")
    
    # Customize the axes and title
    ax.set_title(f'Mean, Minima Mean, and Maxima Mean vs Frequency for {pump_name}')
    ax.set_xlabel('Frequency (f)')
    ax.set_ylabel('Tail Value')
    
    # Show legend
    ax.legend()
    
    # Adjust the layout and show the plot
    plt.tight_layout()
    plt.show()

# Example usage
plot_aggregated_tails(df, 'Н00182642')

#plot_std_dev_vs_mean(df)

# plt.plot(df.iloc[0]['tail'])
# plt.plot(get_maxima(df.iloc[0]['tail']))
# plt.plot(get_minima(df.iloc[0]['tail']))
# plt.show()