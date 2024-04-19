import matplotlib.pyplot as plt
import pandas as pd
import util
import numpy as np

df_05G = pd.read_csv('data/new/4,4-4,0-0,5G.csv', index_col=None, header=None)
df_1G = pd.read_csv('data/new/4,4-4,35-1G.csv', index_col=None, header=None)

# Ensure df_1G has the same number of rows as df_05G by appending rows of zeros
df_05G, df_1G = util.add_missing_rows(df_05G, df_1G)

# численное интегрирование по неотфильтрованному давлению
#df_target_unfiltered_05G = util.calculate_integral(df_05G)
#df_target_unfiltered_1G = util.calculate_integral(df_1G)

v_05G = df_05G.values
print(np.sum(np.isnan(v_05G)))
v_1G = df_1G.values
print(np.sum(np.isnan(v_1G)))

#util.plot_2_ndarrays(v_05G, v_1G)

v_vol_05G = v_05G.cumsum()
v_vol_1G = v_1G.cumsum()

df_05G_filtered = util.interpolate_outliers(df_05G)
df_1G_filtered = util.interpolate_outliers(df_1G)

print(np.sum(np.isnan(df_05G_filtered.values)))
print(np.sum(np.isnan(df_1G_filtered.values)))

v_05G_filtered = util.moving_average_filter(df_05G_filtered, 200, 5000).values
v_1G_filtered = util.moving_average_filter(df_1G_filtered, 200, 5000).values

print(np.sum(np.isnan(v_05G_filtered)))
print(np.sum(np.isnan(v_1G_filtered)))

v_vol_filtered_05G = v_05G_filtered.cumsum()
v_vol_filtered_1G = v_1G_filtered.cumsum()

#plt.plot(v_vol_filtered_05G)
#plt.plot(v_vol_filtered_1G)
plt.subplot(2, 1, 1)
plt.plot(v_05G_filtered)
plt.subplot(2, 1, 2)
plt.plot(v_vol_filtered_05G)
plt.show()