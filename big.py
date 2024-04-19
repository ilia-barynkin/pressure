import pandas as pd
import matplotlib.pyplot as plt

filename = 'data/big.csv'
output_dir = 'data/output'

df = pd.read_csv(filename, delimiter = ',', index_col=None)

def name_columns_by_nominal(pd_df):
    curr_nominal = ''
    column_counter = 0
    for column in pd_df.columns:
        try:
            first_cell = pd_df[column].iloc[0]
            if any(char.isalpha() for char in str(first_cell)):
                curr_nominal = first_cell
                pd_df.drop(index=0, inplace=True)
                column_counter = 1
            else:
                pd_df = pd_df.rename(columns={column: f"{curr_nominal}_{column_counter}"})
                column_counter += 1
        except:
            pd_df = pd_df.rename(columns={column: f"{curr_nominal}_{column_counter}"})
            column_counter += 1
    return pd_df

df = name_columns_by_nominal(df)

df = df.apply(lambda x: x.astype(str).str.replace('[^0-9]', '', regex=True))

# Преобразуем значения в числовые типы
df = df.apply(pd.to_numeric, errors='coerce')
df = df.interpolate()

# Save each column as a separate plot
def save_plt(col):
    plt.figure()
    plt.plot(df[column])
    plt.title(f'Plot of {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.savefig(output_dir + f'/plot_{column}.png')

# Plot each column in df on a separate figure
for column in df.columns:
    save_plt(column)

from scipy import stats
import numpy as np

def remove_outliers(df):
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col][(np.abs(stats.zscore(df[col])) < 3)]
    return df

df_actual = remove_outliers(df_measurements)

df_actual.plot()
plt.show()

from scipy.signal import butter, filtfilt

# Define the high-pass filter
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

# Apply the high-pass filter to each column of the dataframe
def highpass_filter_dataframe(df, cutoff, fs, order=5):
    filtered_df = pd.DataFrame()
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            b, a = butter_highpass(cutoff, fs, order=order)
            filtered_df[column] = filtfilt(b, a, df[column])
    return filtered_df

# Assuming you define your sampling frequency (fs) and cutoff frequency
fs = 100  # Replace with the actual sampling frequency
cutoff = 1  # Replace with the desired cutoff frequency

# Apply the high-pass filter to df_measurements
df_measurements_filtered = highpass_filter_dataframe(df_measurements, cutoff, fs)


from scipy.signal import lfilter, lfilter_zi

def moving_average_filter(df, sampling_rate=200, window_size=5):
    """Applies a moving average filter to all numeric columns in a DataFrame.
    
    Args:
        df: A pandas DataFrame with numeric columns to be filtered.
        sampling_rate: The sampling rate (in Hz) of the data.
        window_size: The size of the moving average window in samples.
    
    Returns:
        A DataFrame with the moving average filter applied to all numeric columns.
    """
    # Coefficients for the moving average filter
    b = np.ones(window_size) / window_size
    a = 1
    
    # Create an empty DataFrame to store the filtered data
    filtered_df = pd.DataFrame(index=df.index)
    
    # Apply the filter to each numeric column
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Initial conditions for the filter
            zi = lfilter_zi(b, a) * df[column].iloc[0]
            # Apply the filter
            filtered_column, _ = lfilter(b, a, df[column], zi=zi)
            filtered_df[column] = filtered_column
    
    return filtered_df

# Apply the moving average filter to df_measurements
df_measurements_filtered = moving_average_filter(df_measurements)


def find_local_extrema(df):
    from scipy.signal import argrelextrema
    
    extrema_df = pd.DataFrame(index=df.index)
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Find indexes of local maxima
            local_max_idx = argrelextrema(df[column].values, np.greater)[0]
            # Find indexes of local minima
            local_min_idx = argrelextrema(df[column].values, np.less)[0]
            
            # Create a Series to store extrema values with the index set to 'Max' or 'Min'
            extrema_series = pd.Series(index=df.index)
            extrema_series.iloc[local_max_idx] = df[column].iloc[local_max_idx]
            extrema_series.iloc[local_min_idx] = df[column].iloc[local_min_idx]
            
            # Store the extrema values in the DataFrame using MultiIndex
            extrema_df[(column, 'Max')] = pd.Series(extrema_series.iloc[local_max_idx], index=local_max_idx)
            extrema_df[(column, 'Min')] = pd.Series(extrema_series.iloc[local_min_idx], index=local_min_idx)
    
    # Drop rows where all elements are NaN since they are not extrema
    extrema_df = extrema_df.dropna(how='all')
    
    # Pivot the DataFrame to have a hierarchical column index (column name, 'Max'/'Min')
    extrema_df.columns = pd.MultiIndex.from_tuples(extrema_df.columns, names=['Column', 'Extrema'])
    
    return extrema_df

# Apply the function to find local extrema for df_measurements_filtered
df_measurements_extrema = find_local_extrema(df_measurements_filtered)
