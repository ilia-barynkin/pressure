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