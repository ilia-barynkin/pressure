import pandas as pd
import matplotlib.pyplot as plt

data_barometer_without_0_5_5_hz = pd.read_csv('data/Интегрированный_барометр_без_0,5_5Гц.csv')
data_barometer_long_before_0_5_5_hz = pd.read_csv('data/Интегрированный_барометр_длинный_ДО_0,5_5Гц_5.csv')
data_barometer_long_after_0_5_5_hz = pd.read_csv('data/Интегрированный_барометр_длинный_ПОСЛЕ_0,5_5Гц_.csv')


plt.plot(data_barometer_long_after_0_5_5_hz)
plt.show()