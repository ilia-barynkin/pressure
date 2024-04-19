import matplotlib.pyplot as plt
import pandas as pd
import util
import numpy as np
from math import e, log

df_05G = pd.read_csv('data/new/4,4-4,0-0,5G.csv', index_col=None, header=None)
df_1G = pd.read_csv('data/new/4,4-4,35-1G.csv', index_col=None, header=None)
df_2G = pd.read_csv('data/new/4,4-4,35-2G.csv', index_col=None, header=None)

#df_05G, df_1G = util.add_missing_rows(df_05G, df_1G)

v_05G = df_05G.to_numpy().flatten()
v_1G = df_1G.to_numpy().flatten()
v_2G = df_2G.to_numpy().flatten()

def corr(v1, v2):
    min_sz = min(v1.size, v2.size)
    return np.corrcoef(v1[0:min_sz], v2[0:min_sz])

corr_05_1 = corr(v_1G, v_05G)
corr_1_2 = corr(v_2G, v_1G)
corr_05_2 = corr(v_2G, v_05G)

print('corr_05_1', corr_05_1)
print('corr_05_2', corr_05_2)
print('corr_1_2', corr_1_2)

v_sum_05G = v_05G.cumsum()
v_sum_1G = v_1G.cumsum()
v_sum_2G = v_2G.cumsum()

v_sum_extended_1G = np.linspace(v_sum_1G.min(), v_sum_1G.max(), v_sum_05G.size)
v_sum_extended_2G = np.linspace(v_sum_2G.min(), v_sum_2G.max(), v_sum_05G.size)

print('corr_sum: ')

corr_sum_05_1 = corr(v_sum_1G, v_sum_05G)
corr_sum_1_2 = corr(v_sum_2G, v_sum_1G)
corr_sum_05_2 = corr(v_sum_2G, v_sum_05G)

max_sum_val = max(v_sum_05G.max(), v_sum_1G.max(), v_sum_2G.max())

plt.plot(v_sum_05G, label='0,5G')
plt.plot(v_sum_1G, label='1G')
plt.plot(v_sum_2G, label='2G')
plt.legend()
plt.show()