import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data_file  = "input.csv"
dta = pd.read_csv("input.csv", index_col=None)
dta = dta.iloc[:, 1]
res = sm.tsa.ARIMA(dta, (20, 1, 6)).fit()
fig, ax = plt.subplots()
ax = dta.iloc[:].plot(ax=ax)
res.plot_predict(50, 100, dynamic=True, ax=ax,
                 plot_insample=False)
plt.show()