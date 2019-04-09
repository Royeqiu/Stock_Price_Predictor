import talib
import random
import math
from sklearn.linear_model import LogisticRegression


def MA_model(openPrice, close_data, capital, alpha=0.003, beta=0.001):
    alpha = openPrice * alpha
    beta = openPrice * beta
    close_data_SMA = talib.SMA(close_data, 11)
    if alpha is None or beta is None:
        return 0
    elif openPrice - alpha > close_data_SMA[len(close_data_SMA) - 1]:

        return 1

    elif openPrice + beta < close_data_SMA[len(close_data_SMA) - 1]:

        return -1

    else:
        return 0


def KD_model(high, low, close, alpha, beta, capital):
    Stochastic_K, Stochastic_D = talib.STOCH(high, low, close, fastk_period=11, slowk_period=9, slowk_matype=0,
                                             slowd_period=9, slowd_matype=0)
    if (Stochastic_K[len(Stochastic_K) - 1] < alpha and Stochastic_K[len(Stochastic_K) - 1] - Stochastic_D[
        len(Stochastic_D) - 1] > 10) or Stochastic_K[len(Stochastic_K) - 1] < 30:

        return 1


    elif (Stochastic_K[len(Stochastic_K) - 1] > beta and Stochastic_D[len(Stochastic_D) - 1] - Stochastic_K[
        len(Stochastic_K) - 1] > 5) or Stochastic_K[len(Stochastic_K) - 1] > 70:

        return -1

    else:
        return 0


def RSI_model(close, openPrice, capital):
    RSI = talib.RSI(close, 11)  # RSI
    rsi_Index = RSI[len(RSI) - 1]
    if rsi_Index <= 15:
        if capital != 0:
            RSI_model.buy_price = openPrice
        return 1

    elif rsi_Index > 70:
        if RSI_model.buy_price < openPrice:
            return -1
        else:
            return 0
    elif len(RSI) > 10:
        if RSI[len(RSI) - 1] > RSI[len(RSI) - 2] and RSI[len(RSI) - 2] > RSI[len(RSI) - 3] and rsi_Index < 70:
            if capital != 0:
                RSI_model.buy_price = openPrice
            return 1

        elif RSI[len(RSI) - 1] < RSI[len(RSI) - 2] and RSI[len(RSI) - 2] < RSI[len(RSI) - 3] and rsi_Index > 20:
            if RSI_model.buy_price < openPrice:
                return -1
            else:
                return 0
        else:
            return 0
    else:
        return 0


def ML_model(open, close, high, low, volume, openPrice):
    logreg = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', max_iter=6000)
    sma_vec = talib.SMA(close, 11)
    sma_vol_vec = talib.SMA(volume, 11)
    k_vec, d_vec = talib.STOCH(high, low, close, fastk_period=11, slowk_period=9,
                               slowk_matype=0,
                               slowd_period=9, slowd_matype=0)
    rsi_vec = talib.RSI(close, 11)  # RSI
    train_x = []
    train_y = []
    for i, data in enumerate(sma_vec):
        if i == len(sma_vec) - 1:
            break
        if math.isnan(data) or math.isnan(k_vec[i]) or math.isnan(d_vec[i]) or math.isnan(rsi_vec[i]):
            continue
        sma_index = sma_vec[i]
        rsi_index = rsi_vec[i]
        k = k_vec[i]
        d = d_vec[i]
        subkd = k - d
        sma_vol = volume[i]
        last_distance = open[i] - close[i]
        high_dis = high[i] - open[i]
        low_dis = open[i] - low[i]
        tmp_x = []
        tmp_x.append(open[i + 1] - sma_index)
        tmp_x.append(rsi_index)
        tmp_x.append(k)
        tmp_x.append(d)
        tmp_x.append(subkd)
        tmp_x.append(sma_vol)
        tmp_x.append(last_distance)
        tmp_x.append(open[i + 1])
        tmp_x.append(high_dis)
        tmp_x.append(low_dis)
        train_x.append(tmp_x)
        if close[i + 1] > close[i]:
            train_y.append(1)
        elif close[i + 1] < close[i]:
            train_y.append(2)
        else:
            train_y.append(0)
    logreg.fit(train_x, train_y)
    last_sma = openPrice - sma_vec[len(sma_vec) - 1]
    last_rsi = rsi_vec[len(rsi_vec) - 1]
    last_k = k_vec[len(k_vec) - 1]
    last_d = d_vec[len(d_vec) - 1]
    subkd = last_k - last_d
    sma_vol = volume[len(volume) - 1]
    last_distance = open[len(open) - 1] - close[len(close) - 1]
    high_dis = high[len(high) - 1] - open[len(open) - 1]
    low_dis = open[len(open) - 1] - low[len(low) - 1]
    next = [last_sma, last_rsi, last_k, last_d, subkd, sma_vol, last_distance, openPrice, high_dis, low_dis]
    y = logreg.predict([next])
    return y[0]


def cal_val(action, capital, stock, openPrice):
    if action == 1:
        stock = (capital - 100) / openPrice
        capital = 0
    elif action == -1:
        capital = stock * openPrice - 100
        stock = 0
    value = stock * openPrice - 100 + capital

    return value


def myStrategy(dailyOhlcvFile, minutelyOhlcvFile, openPrice):
    high = dailyOhlcvFile['high']
    open = dailyOhlcvFile['open']
    low = dailyOhlcvFile['low']
    close = dailyOhlcvFile['close']
    volume = dailyOhlcvFile['volume']
    RSI_action = RSI_model(close, openPrice, myStrategy.RSI_capital)
    KD_action = KD_model(high, low, close, alpha=20, beta=80, capital=myStrategy.KD_capital)
    MA_action = MA_model(openPrice, close, myStrategy.MA_capital)
    ML_action = ML_model(open, close, high, low, volume, openPrice)
    if ML_action == 2:
        ML_action = -1
    if MA_action == 1 and myStrategy.MA_capital != 0:
        myStrategy.MA_stock = (myStrategy.MA_capital - 100) / openPrice
        myStrategy.MA_capital = 0
    elif MA_action == -1 and myStrategy.MA_stock != 0:
        myStrategy.MA_capital = myStrategy.MA_stock * openPrice - 100
        myStrategy.MA_stock = 0
    MA_value = myStrategy.MA_stock * openPrice - 100 + myStrategy.MA_capital

    if KD_action == 1 and myStrategy.KD_capital != 0:
        myStrategy.KD_stock = (myStrategy.KD_capital - 100) / openPrice
        myStrategy.KD_capital = 0
    elif KD_action == -1 and myStrategy.KD_stock != 0:
        myStrategy.KD_capital = myStrategy.KD_stock * openPrice - 100
        myStrategy.KD_stock = 0
    KD_value = myStrategy.KD_stock * openPrice - 100 + myStrategy.KD_capital
    if RSI_action == 1 and myStrategy.RSI_capital != 0:
        myStrategy.RSI_stock = (myStrategy.RSI_capital - 100) / openPrice
        myStrategy.RSI_capital = 0
    elif RSI_action == -1 and myStrategy.RSI_stock != 0:
        myStrategy.RSI_capital = myStrategy.RSI_stock * openPrice - 100
        myStrategy.RSI_stock = 0
    RSI_value = myStrategy.RSI_stock * openPrice - 100 + myStrategy.RSI_capital

    if ML_action == 1 and myStrategy.ML_capital != 0:
        myStrategy.ML_stock = (myStrategy.ML_capital - 100) / openPrice
        myStrategy.ML_capital = 0
    elif ML_action == -1 and myStrategy.ML_stock != 0:
        myStrategy.ML_capital = myStrategy.ML_stock * openPrice - 100
        myStrategy.ML_stock = 0
    ML_value = myStrategy.ML_stock * openPrice - 100 + myStrategy.ML_capital
    doma = sum([abs(KD_value - 500000), abs(RSI_value - 500000), abs(ML_value - 500000)])
    value_list = [(KD_value - 500000) / doma, (RSI_value - 500000) / doma, (ML_value - 500000) / doma]
    weight_list = softmax(value_list)
    action_board = {0: 0.0, 1: 0.0, -1: 0.0}
    action_board[KD_action] += weight_list[0]
    action_board[RSI_action] += weight_list[1]
    action_board[ML_action] += weight_list[2]
    action_score=[action_board[0],action_board[1],action_board[-1]]
    max_score = max(action_score)
    action = action_score.index(max_score)
    if action == 2:
        action = -1

    if action == 1 and myStrategy.capital != 0:
        myStrategy.stock = (myStrategy.capital - 100) / openPrice
        myStrategy.capital = 0
    elif action == -1 and myStrategy.stock != 0:
        myStrategy.capital = myStrategy.stock * openPrice - 100
        myStrategy.stock = 0
    else:
        action = 0

    if KD_action!= 0:
        action=KD_action
    else:
        action = ML_action


    if KD_action == 1 and myStrategy.KD_capital != 0:
        myStrategy.KD_stock = (myStrategy.KD_capital - 100) / openPrice
        myStrategy.KD_capital = 0
    elif KD_action == -1 and myStrategy.KD_stock != 0:
        myStrategy.KD_capital = myStrategy.KD_stock * openPrice - 100
        myStrategy.KD_stock = 0
    KD_value = myStrategy.KD_stock * openPrice - 100 + myStrategy.KD_capital
    if RSI_action == 1 and myStrategy.RSI_capital != 0:
        myStrategy.RSI_stock = (myStrategy.RSI_capital - 100) / openPrice
        myStrategy.RSI_capital = 0
    elif RSI_action == -1 and myStrategy.RSI_stock != 0:
        myStrategy.RSI_capital = myStrategy.RSI_stock * openPrice - 100
        myStrategy.RSI_stock = 0
    RSI_value = myStrategy.RSI_stock * openPrice - 100 + myStrategy.RSI_capital
    if ML_action == 1 and myStrategy.ML_capital != 0:
        myStrategy.ML_stock = (myStrategy.ML_capital - 100) / openPrice
        myStrategy.ML_capital = 0
    elif ML_action == -1 and myStrategy.ML_stock != 0:
        myStrategy.ML_capital = myStrategy.ML_stock * openPrice - 100
        myStrategy.ML_stock = 0

    ML_value = myStrategy.ML_stock * openPrice - 100 + myStrategy.ML_capital
    value = myStrategy.stock * openPrice - 100 + myStrategy.capital

    return action


def softmax(value_list):
    total_value = 0
    weight_list = []
    for value in value_list:
        total_value += math.exp(value)

    for value in value_list:
        weight_list.append(math.exp(value) / total_value)
    return weight_list


RSI_model.buy_price = 0
myStrategy.capital = 500000
myStrategy.stock = 0
myStrategy.MA_capital = 500000
myStrategy.MA_stock = 0
myStrategy.RSI_capital = 500000
myStrategy.RSI_stock = 0
myStrategy.KD_capital = 500000
myStrategy.KD_stock = 0
myStrategy.ML_capital = 500000
myStrategy.ML_stock = 0
ML_model.buy_price = 0