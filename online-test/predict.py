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
# from sklearn.preprocessing import PolynomialFeatures

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

def rise_ask(df,name='last_price'):
    h = 99
    Ask1 = np.array( df['%s'%name] )
    timestamp_time_second = df['tx_id']
    before_time = len(df.index)/2
    rise_ratio = []
    Ask1[Ask1 == 0] = np.mean(Ask1)
    rise_ratio = []
    index = np.where(timestamp_time_second >= before_time)[0][0]
    #open first before_time mins
    for i in range(0,index,1):
        rise_ratio_ =(Ask1[i] - Ask1[0])*(1.0)*100 / Ask1[0]
        rise_ratio.append(rise_ratio_)
    for i in range(index,len(Ask1),1):
        # index_start = np.where(timestamp_time_second[:i].astype('float') >= timestamp_time_second[i].astype('float') - before_time)[0][0]
        index_start = 0
        rise_ratio_ = round((Ask1[i] - Ask1[index_start])*(1.0)*100/Ask1[index_start],5)
        rise_ratio.append(rise_ratio_)

    df['rise_ask'] = rise_ratio
    return df

def STATS(df,name='last_price',method='mean'):
    h = 99
    Ask1 = np.array( df['%s'%name] )
    timestamp_time_second = df['tx_id']
    INDEX=[]
    for i in range(0,len(Ask1)):
        if i<h:
            last = Ask1[:(i+1)]
        else:
            last = Ask1[(i-h):(i+1)]
        # 均值
        if method=='mean':
            rise_ratio_ = round( np.mean(last) ,2 )
            INDEX.append(np.mean(last))
        elif method=='max':
            rise_ratio_ = round( np.max(last) ,2 )
            INDEX.append(np.mean(last))
        elif method=='min':
            rise_ratio_ = round( np.min(last) ,2 )
            INDEX.append(np.mean(last))
        elif method=='max-min':
            rise_ratio_ = round( np.max(last) - np.min(last) ,2 )
            INDEX.append(np.mean(last))
    df['%s'%name +'_'+'%s'% method] = INDEX
    return df


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

#     # 当日开盘价，开盘交易量
#     data['first_price'] = data['last_price'].iloc[0]
#     data['first_qty'] = data['qty'].iloc[0]


    # ma趋势指标
    for i in range(5, 100, 25):
        data['price_ma' + '%s' % i] = rolling_mean_(df=data, name='last_price', tw=i)
        data['qty_ma' + '%s' % i] = rolling_mean_(df=data, name='qty', tw=i)

        data['ask1_ma' + '%s' % i] = rolling_mean_(df=data, name='best_ask_price_1', tw=i)
        data['ask2_ma' + '%s' % i] = rolling_mean_(df=data, name='best_ask_price_2', tw=i)
        data['ask3_ma' + '%s' % i] = rolling_mean_(df=data, name='best_ask_price_3', tw=i)
        # data['ask4_ma'+'%s'%i] = rolling_mean_(df=data,name='best_ask_price_4',tw=i)
        # data['ask5_ma'+'%s'%i] = rolling_mean_(df=data,name='best_ask_price_5',tw=i)
        data['bid1_ma' + '%s' % i] = rolling_mean_(df=data, name='best_bid_price_1', tw=i)
        data['bid2_ma' + '%s' % i] = rolling_mean_(df=data, name='best_bid_price_2', tw=i)
        data['bid3_ma' + '%s' % i] = rolling_mean_(df=data, name='best_bid_price_3', tw=i)
        # data['bid4_ma'+'%s'%i] = rolling_mean_(df=data,name='best_bid_price_4',tw=i)
        # data['bid5_ma'+'%s'%i] = rolling_mean_(df=data,name='best_bid_price_5',tw=i)

        data['ask_size1_ma' + '%s' % i] = rolling_mean_(df=data, name='best_ask_size_1', tw=i)
        data['ask_size2_ma' + '%s' % i] = rolling_mean_(df=data, name='best_ask_size_2', tw=i)
        data['ask_size3_ma' + '%s' % i] = rolling_mean_(df=data, name='best_ask_size_3', tw=i)
        # data['ask_size4_ma'+'%s'%i] = rolling_mean_(df=data,name='best_ask_size_4',tw=i)
        # data['ask_size5_ma'+'%s'%i] = rolling_mean_(df=data,name='best_ask_size_5',tw=i)
        data['bid_size1_ma' + '%s' % i] = rolling_mean_(df=data, name='best_bid_size_1', tw=i)
        data['bid_size2_ma' + '%s' % i] = rolling_mean_(df=data, name='best_bid_size_2', tw=i)
        data['bid_size3_ma' + '%s' % i] = rolling_mean_(df=data, name='best_bid_size_3', tw=i)
        # data['bid_size4_ma'+'%s'%i] = rolling_mean_(df=data,name='best_bid_size_4',tw=i)
        # data['bid_size5_ma'+'%s'%i] = rolling_mean_(df=data,name='best_bid_size_5',tw=i)

        data['spread' + '%s' % i] = rolling_mean_(df=data, name='spread', tw=i)
        data['spread_weight' + '%s' % i] = rolling_mean_(df=data, name='spread_weight', tw=i)
        data['deapth' + '%s' % i] = rolling_mean_(df=data, name='deapth', tw=i)

        data['price_ma' + '%s' % i] = rolling_std_(df=data, name='last_price', tw=i)
        data['qty_ma' + '%s' % i] = rolling_std_(df=data, name='qty', tw=i)

        data['ask1_ma' + '%s' % i] = rolling_std_(df=data, name='best_ask_price_1', tw=i)
        data['ask2_ma' + '%s' % i] = rolling_std_(df=data, name='best_ask_price_2', tw=i)
        data['ask3_ma' + '%s' % i] = rolling_std_(df=data, name='best_ask_price_3', tw=i)
        # data['ask4_ma'+'%s'%i] = rolling_std_(df=data,name='best_ask_price_4',tw=i)
        # data['ask5_ma'+'%s'%i] = rolling_std_(df=data,name='best_ask_price_5',tw=i)
        data['bid1_ma' + '%s' % i] = rolling_std_(df=data, name='best_bid_price_1', tw=i)
        data['bid2_ma' + '%s' % i] = rolling_std_(df=data, name='best_bid_price_2', tw=i)
        data['bid3_ma' + '%s' % i] = rolling_std_(df=data, name='best_bid_price_3', tw=i)
        # data['bid4_ma'+'%s'%i] = rolling_std_(df=data,name='best_bid_price_4',tw=i)
        # data['bid5_ma'+'%s'%i] = rolling_std_(df=data,name='best_bid_price_5',tw=i)

        data['ask_size1_ma' + '%s' % i] = rolling_std_(df=data, name='best_ask_size_1', tw=i)
        data['ask_size2_ma' + '%s' % i] = rolling_std_(df=data, name='best_ask_size_2', tw=i)
        data['ask_size3_ma' + '%s' % i] = rolling_std_(df=data, name='best_ask_size_3', tw=i)
        # data['ask_size4_ma'+'%s'%i] = rolling_std_(df=data,name='best_ask_size_4',tw=i)
        # data['ask_size5_ma'+'%s'%i] = rolling_std_(df=data,name='best_ask_size_5',tw=i)
        data['bid_size1_ma' + '%s' % i] = rolling_std_(df=data, name='best_bid_size_1', tw=i)
        data['bid_size2_ma' + '%s' % i] = rolling_std_(df=data, name='best_bid_size_2', tw=i)
        data['bid_size3_ma' + '%s' % i] = rolling_std_(df=data, name='best_bid_size_3', tw=i)
        # data['bid_size4_ma'+'%s'%i] = rolling_std_(df=data,name='best_bid_size_4',tw=i)
        # data['bid_size5_ma'+'%s'%i] = rolling_std_(df=data,name='best_bid_size_5',tw=i)

        data['spread' + '%s' % i] = rolling_std_(df=data, name='spread', tw=i)
        data['spread_weight' + '%s' % i] = rolling_std_(df=data, name='spread_weight', tw=i)
        data['deapth' + '%s' % i] = rolling_std_(df=data, name='deapth', tw=i)



    # poly = PolynomialFeatures(interaction_only=True)
    # feature_new = poly.fit_transform(data.loc[:, 'best_bid_price_1':'qty'])
    # data = data.join(pd.DataFrame(feature_new))

    data = data.reset_index(drop=True)
    
    data = data.reset_index(drop=True)
    data = data.groupby('day').apply(lambda x: rise_ask(x, name='last_price'))
    data = data.groupby('day').apply(lambda x: STATS(x, name='best_bid_price_1', method='max-min'))
    data = data.groupby('day').apply(lambda x:STATS(x,name='best_bid_price_2',method='max-min') )
    data = data.groupby('day').apply(lambda x:STATS(x,name='best_ask_price_1',method='max-min') )
    data = data.groupby('day').apply(lambda x:STATS(x,name='best_ask_price_2',method='max-min') )

#     data = data.fillna(0)
#     data['pan'] = data.apply(lambda x: cal_pan(int(x.iloc[0]), data, key='pan'))
#     data = data.fillna(0)
#     data['pan_vol'] = data.apply(lambda x: cal_pan(int(x.iloc[0]), data, key='pan_vol'))
    data = data.fillna(0)
    data = data.reset_index(drop=True)
    data2 = data.groupby('day').apply(rolling)
    # data2.dropna(axis=1,how='all',inplace=True)

    data2.drop(['day'],axis=1,inplace=True)
    
    return data2

def predict(tick_data):
    x_train = mk_ratio(tick_data.astype('float'), ['qty', 'best_bid_price_1'])
    tick_data_pro = submit(x_train)
    
    tick_data_last = tick_data_pro.iloc[[-1]]

    y_sub_all = np.zeros([len(tick_data_last), 4])
    for i in range(4):
    
#         for fold in range(5):
        xgb_model_loaded = pickle.load(open("xgb_regression_%s"% 1+'_%s'%i+"_100+feature.pkl", "rb"))
        data_test = xgb.DMatrix(tick_data_last)
        y_sub_all[:, i] = xgb_model_loaded.predict(data_test)
    result = np.argmax(y_sub_all, axis=1)

    return result


if __name__ == '__main__':
    predict()



# if __name__ == '__main__':
#     for _ in range(3):
#         print(predict(""))
