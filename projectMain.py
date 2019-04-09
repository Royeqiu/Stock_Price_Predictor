import numpy as np
import pandas as pd

from myStrategy import myStrategy

dailyOhlcv = pd.read_csv('data/ohlc_daily.csv')
evalDays = 140
action = np.zeros((evalDays,1))
openPricev = dailyOhlcv["open"].tail(evalDays).values
for ic in range(evalDays,0,-1):
    dailyOhlcvFile = dailyOhlcv.head(len(dailyOhlcv)-ic)
    dateStr = dailyOhlcvFile.iloc[-1,0]
    minutelyOhlcvFile = None
    action[evalDays-ic] = myStrategy(dailyOhlcvFile,minutelyOhlcvFile,openPricev[evalDays-ic])
