#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Date    : 2021-06-01 20:03:39
 @Author  : Li Huaijun (lihuaijun@cmbchina.com)
 @Version : $Id$
 @Desc:
"""

from random import choice
import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import xgboost as xgb


def sin_ratio(df, col):
    col_ratio = pd.DataFrame(df.groupby(['day', 'tx_id']).sum()[col])
    result = pd.merge(df, col_ratio, how='left', on=['day', 'tx_id'])
    result['ratio'] = result[col + '_x'] / result[col + '_y']
    cols = result.columns
    for col in cols[2:-2]:
        result[col] = result[col] * result['ratio']
    result_new = result.iloc[:, :-2]
    result_new.columns = df.columns
    return result_new


def mk_ratio(df, cols):
    ratio_ = pd.DataFrame(df.groupby(['day', 'tx_id']).sum()[cols])
    result = pd.merge(df, ratio_, how='left', on=['day', 'tx_id'])
    for col in cols:
        result[col + '_ratio'] = result[col + '_x'] / result[col + '_y']
    cols_ratio = []
    for col in cols:
        cols_ratio.append(col + '_ratio')
    result['ratio'] = result[cols_ratio].mean(axis=1)
    result_new = result.iloc[:, :24]
    result_new.columns = df.columns
    for col in result_new.columns[2:]:
        result_new[col] = result_new[col] * result['ratio']
    result_new = result_new.groupby(['day', 'tx_id']).sum()

    return pd.DataFrame(result_new.groupby(['day', 'tx_id']).sum().reset_index()).reset_index(drop=True)


#########################################################################
# 为bid与offer的volume变动量之差（标准化后）衡量交易意愿强弱
# https://zhuanlan.zhihu.com/p/296523150
def order_imbalance(x, df, key):
    # print(x)
    df = df.iloc[:x, :]
    # print(df)
    deltavb = max(df['best_bid_price_1'].iloc[-1] - df['best_bid_price_1'].iloc[-2],
                  -df['best_bid_price_1'].iloc[-1] + df['best_bid_price_1'].iloc[-2], 0)
    deltava = max(df['best_ask_price_1'].iloc[-1] - df['best_ask_price_1'].iloc[-2],
                  -df['best_ask_price_1'].iloc[-1] + df['best_ask_price_1'].iloc[-2], 0)
    rho = (deltavb - deltava) / (deltavb + deltava)
    mid = (df['best_bid_price_1'].iloc[-1] + df['best_ask_price_1'].iloc[-1]) / 2
    tp = max(
        (df['last_price'].iloc[-1] - df['last_price'].iloc[-2]) / (300 * (df['qty'].iloc[-1] - df['qty'].iloc[-2])), 0)
    Mid_price_basis = tp - mid
    if key == 'deltavb':
        return deltavb
    elif key == 'deltava':
        return deltavb
    elif key == 'rho':
        return rho
    elif key == 'mid':
        return mid
    elif key == 'tp':
        return tp


def rolling_mean_(df, name, tw):
    df['%s' % name + '%s' % tw] = df['%s' % name].rolling(window=tw).mean()
    return df['%s' % name + '%s' % tw]


def rolling_std_(df, name, tw):
    df['%s' % name + '%s' % tw] = df['%s' % name].rolling(window=tw).std()
    return df['%s' % name + '%s' % tw]


# 最大回撤，对成交价、所有买卖价都计算一遍。最高价格到最低价格之间的距离
# 最大亏损：对成交价、所有买卖价都计算一遍。初始价格到最低价格之间的距离
# 最大收益：对成交价、所有买卖价都计算一遍。初始价格到最高价格之间的距离, 最高点和开盘价的差异
def MaxDrawdown(x, portfolio, tw):
    if x < tw + 1:
        return 0
    data = portfolio.iloc[x - tw:x].values
    end = np.argmax((np.maximum.accumulate(data) - data))
    begin = np.argmax(data[:end])
    return ((data[begin] - data[end]) / data[begin])


def MaxLoss(x, portfolio, tw):
    # print(type(x),type(tw))
    # print(portfolio)
    if x < tw + 1:
        return 0
    data = portfolio.iloc[x - tw:x].values
    end = np.argmin(np.minimum.accumulate(data))
    begin = np.argmin(data[0])
    return (-1) * (data[end] - data[begin]) / data[begin]


def MaxProfit(x, portfolio, tw):
    if x < tw + 1:
        return 0
    data = portfolio.iloc[x - tw:x].values
    end = np.argmin(np.maximum.accumulate(data))
    begin = np.argmin(data[0])
    return (-1) * (data[end] - data[begin]) / data[begin]


# 当前点与最高点与次高点差异
def highdff(x, portfolio, tw):
    if x < tw + 1:
        return 0
    data = portfolio.iloc[x - tw:x].values
    begin = np.argmax(data)
    end = np.argmin(data[-1])
    return (-1) * (data[end] - data[begin]) / data[begin]


def rolling(df, time_window=900):
    df = df.reset_index(drop=True)
    df['maxloss'] = df['last_price'].rolling(window=time_window).apply(
        lambda x: MaxLoss(int(x.index[0]), df['last_price'], time_window))
    df['maxdd'] = df['last_price'].rolling(window=time_window).apply(
        lambda x: MaxDrawdown(int(x.index[0]), df['last_price'], time_window))
    df['maxprofit'] = df['last_price'].rolling(window=time_window).apply(
        lambda x: MaxDrawdown(int(x.index[0]), df['last_price'], time_window))

    return df


def rise_ask(df, name='last_price'):
    h = 99
    Ask1 = np.array(df['%s' % name])
    timestamp_time_second = df['tx_id']
    before_time = len(df.index) / 2
    rise_ratio = []
    Ask1[Ask1 == 0] = np.mean(Ask1)
    rise_ratio = []
    index = np.where(timestamp_time_second >= before_time)[0][0]
    # open first before_time mins
    for i in range(0, index, 1):
        rise_ratio_ = (Ask1[i] - Ask1[0]) * (1.0) * 100 / Ask1[0]
        rise_ratio.append(rise_ratio_)
    for i in range(index, len(Ask1), 1):
        # index_start = np.where(timestamp_time_second[:i].astype('float') >= timestamp_time_second[i].astype('float') - before_time)[0][0]
        index_start = 0
        rise_ratio_ = round((Ask1[i] - Ask1[index_start]) * (1.0) * 100 / Ask1[index_start], 5)
        rise_ratio.append(rise_ratio_)

    df['rise_ask'] = rise_ratio
    return df


def STATS(df, name='last_price', method='mean'):
    h = 99
    Ask1 = np.array(df['%s' % name])
    timestamp_time_second = df['tx_id']
    INDEX = []
    for i in range(0, len(Ask1)):
        if i < h:
            last = Ask1[:(i + 1)]
        else:
            last = Ask1[(i - h):(i + 1)]
        # 均值
        if method == 'mean':
            rise_ratio_ = round(np.mean(last), 2)
            INDEX.append(np.mean(last))
        elif method == 'max':
            rise_ratio_ = round(np.max(last), 2)
            INDEX.append(np.mean(last))
        elif method == 'min':
            rise_ratio_ = round(np.min(last), 2)
            INDEX.append(np.mean(last))
        elif method == 'max-min':
            rise_ratio_ = round(np.max(last) - np.min(last), 2)
            INDEX.append(np.mean(last))
    df['%s' % name + '_' + '%s' % method] = INDEX
    return df


# 内盘	主动性卖盘，即成交价在买入挂单价的累积成交量;也就是说，期货交易在买入价成交，成交价为申买价，说明抛盘比较踊跃。
# 当卖方主动成交时（买价成交），t1最新价（价位）=t0某一买价，这一买价为大于卖单限价的最低价，视为卖方主动成交；
# 外盘	主动性买盘，即成交价在卖出挂单价的累积成交量;也就是说，期货交易在卖出价成交，成交价为申卖价，说明买盘比较踊跃。
# 当买方主动成交时（卖价成交）,t1最新价（价位）=t0某一卖价，这一卖价为小于买单限价的最高价，视为买方主动成交。
# %%
def cal_pan(x, df, key):
    df = df.reset_index(drop=True)
    df = df.iloc[:x, :]
    # print(df)
    if x > 1:
        if (df['last_price'].iloc[-1] <= df['best_bid_price_1'].iloc[-2] + 5) or (
                df['last_price'].iloc[-1] >= df['best_bid_price_1'].iloc[-2] - 5):
            pan = 1
            pan_vol = df['best_bid_size_1'].iloc[-1] + df['best_bid_size_2'].iloc[-1] + df['best_bid_size_3'].iloc[-1] + \
                      df['best_bid_size_4'].iloc[-1] + df['best_bid_size_5'].iloc[-1]
        elif (df['last_price'].iloc[-1] <= df['best_ask_price_1'].iloc[-2] + 5) or (
                df['last_price'].iloc[-1] >= df['best_ask_price_1'].iloc[-2] - 5):
            # df['last_price'].iloc[-1] == df['best_ask_price_1'].iloc[-2]:
            pan = -1
            pan_vol = df['best_ask_size_1'].iloc[-1] + df['best_ask_size_2'].iloc[-1] + df['best_ask_size_3'].iloc[-1] + \
                      df['best_ask_size_4'].iloc[-1] + df['best_ask_size_5'].iloc[-1]
        else:
            pan, pan_vol = 0, 0
    else:
        pan, pan_vol = 0, 0
    if key == 'pan':
        return pan
    elif key == 'pan_vol':
        return pan_vol


def submit(data):
    data.drop_duplicates(subset=['day', 'tx_id'], keep='last', inplace=True)
    # bid-ask spread
    data['spread'] = data['best_bid_price_1'] - data['best_ask_price_1']
    data['spread_weight'] = data['best_bid_price_1'] * data['best_bid_size_1'] / data['qty'] - \
                            data['best_ask_price_1'] * data['best_bid_size_1'] / data['qty']

    # 深度比
    data['deapth'] = (data['best_bid_price_1'] * data['best_bid_size_1'] + data['best_bid_price_2'] * data[
        'best_bid_size_2'] + \
                      data['best_bid_price_3'] * data['best_bid_size_3']) / (
                             data['best_ask_price_1'] * data['best_bid_size_1'] +
                             data['best_ask_price_2'] * data['best_ask_size_2'] +
                             data['best_ask_price_3'] * data['best_ask_size_3'])

    # 当日开盘价，开盘交易量
    # data['first_price'] = data['last_price'].iloc[0]
    # data['first_qty'] = data['qty'].iloc[0]

    # 1.tx_id
    data = data.reset_index(drop=True)
    # 2.rise_ask
    data = data.groupby('day').apply(lambda x: rise_ask(x, name='best_bid_price_1'))
    data = data.reset_index(drop=True)
    # 3.maxdd 4.maxloss 5.maxprofit
    data = data.groupby('day').apply(rolling)
    # 6.
    # qty80
    # 7.qty_ma80
    data['qty_ma80'] = rolling_mean_(df=data, name='qty', tw=80)
    # 8. deapth_80
    data['deapth80'] = rolling_mean_(df=data, name='deapth', tw=80)
    # 9.  'best_bid_size_5', 10.'best_ask_size_5',
    # 11. 'bid_size3_ma80'
    data['bid_size3_ma80'] = rolling_mean_(df=data, name='best_bid_size_3', tw=80)
    # 12. best_bid_size_380
    data['best_bid_size_380'] = rolling_mean_(df=data, name='best_bid_size_3', tw=80)
    # 13. best_bid_size_4
    # 14. best_ask_size_180
    data['best_ask_size_180'] = rolling_mean_(df=data, name='best_ask_size_1', tw=80)
    # 15. ask_size3_ma80
    data['ask_size3_ma80'] = rolling_mean_(df=data, name='best_ask_size_3', tw=30)
    # 16. bid_size2_ma80
    data['bid_size2_ma80'] = rolling_mean_(df=data, name='best_bid_size_2', tw=80)
    # 17. spread55
    data['spread55'] = rolling_mean_(df=data, name='spread', tw=55)
    # 18. best_ask_size_4
    # 19.best_bid_price_180
    data['best_bid_price_180'] = rolling_mean_(df=data, name='best_bid_price_1', tw=80)
    # 20. qty55
    # 21. spread30
    data['spread30'] = rolling_mean_(df=data, name='spread', tw=30)
    # 22. best_bid_size_280
    data['best_bid_size_280'] = rolling_mean_(df=data, name='best_bid_size_2', tw=80)
    # 23. bid_size1_ma80
    data['bid_size1_ma80'] = rolling_mean_(df=data, name='best_bid_size_1', tw=80)
    # 24. best_ask_size_280
    data['best_ask_size_280'] = rolling_mean_(df=data, name='best_ask_size_2', tw=80)
    # 25. best_ask_size_180
    data['best_ask_size_180'] = rolling_mean_(df=data, name='best_ask_size_1', tw=80)
    # 26. ask_size2_ma80
    data['ask_size2_ma80'] = rolling_mean_(df=data, name='best_ask_size_2', tw=80)

    data['ask_size1_ma80'] = rolling_mean_(df=data, name='best_ask_size_1', tw=80)

    # 27.qty_ma55
    data['qty_ma55'] = rolling_mean_(df=data, name='qty', tw=55)
    # 28. best_bid_price_1
    # 29. deapth_55
    data['deapth55'] = rolling_mean_(df=data, name='deapth', tw=55)
#     # 30. qty30

#     # 31. bid_size3_ma55
#     data['bid_size3_ma55'] = rolling_mean_(df=data, name='best_bid_size_3', tw=55)
#     # 32. best_bid_size_355
#     data['best_ask_size_355'] = rolling_mean_(df=data, name='best_ask_size_3', tw=55)
#     # 33. 'bid_size3_ma30'
#     data['bid_size3_ma30'] = rolling_mean_(df=data, name='best_bid_size_3', tw=30)
#     # 34. ask_size3_ma55'
#     data['ask_size3_ma55'] = rolling_mean_(df=data, name='best_ask_size_3', tw=55)
#     # 35. best_ask_size_3
#     # 36. best_ask_size_355',
#     data['best_ask_size_355'] = rolling_mean_(df=data, name='best_ask_size_3', tw=55)
#     # 37. ask_size3_ma30
#     data['ask_size3_ma30'] = rolling_mean_(df=data, name='best_ask_size_3', tw=30)
#     # 38. 'best_ask_size_3'



#     data['last_price80'] = rolling_mean_(df=data, name='last_price', tw=80)
#     # ask_size1_ma30
#     # ask_size2_ma30
#     # best_bid_size_130
#     # best_ask_size_130
#     #  best_bid_size_1
#     data['last_price55'] = rolling_mean_(df=data, name='last_price', tw=55)
#     data['last_price30'] = rolling_mean_(df=data, name='last_price', tw=30)
#     # bid_size3_ma5
#     # best_ask_size_1
#     # ask_size3_ma5
#     # maxprofit
#     # ask_size2_ma5
#     # bid_size2_ma5


    data = data.reset_index(drop=True)
    data = data.groupby('day').apply(lambda x: STATS(x, name='best_bid_price_1', method='max-min'))
    data = data.groupby('day').apply(lambda x: STATS(x, name='best_ask_price_1', method='max-min'))

    data = data.fillna(0)
    data['pan'] = data.apply(lambda x: cal_pan(int(x.iloc[0]), data, key='pan'))
    data = data.fillna(0)
    data['pan_vol'] = data.apply(lambda x: cal_pan(int(x.iloc[0]), data, key='pan_vol'))
    data = data.fillna(0)

    # data.dropna(axis=1, how='all', inplace=True)
    data.drop(['day'],axis=1,inplace=True)
    return data


def predict(tick_data):
    x_train = mk_ratio(tick_data.astype('float'), ['qty', 'best_bid_price_1'])
    tick_data_pro = submit(x_train)
    
    tick_data_last = tick_data_pro.iloc[[-1]]

    y_sub_all = np.zeros([len(tick_data_last), 4])
    for i in range(4):
    
#    for fold in range(5):
        xgb_model_loaded = pickle.load(open("xgb_regression_%s"%i+"_55_2feature_lr2_sub_all.pkl", "rb"))
        data_test = xgb.DMatrix(tick_data_last)
        y_sub_all[:, i] = xgb_model_loaded.predict(data_test)
    result = np.argmax(y_sub_all, axis=1)

    return result


if __name__ == '__main__':
    predict()