#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Date    : 2021-06-01 19:56:51
 @Author  : Li Huaijun (lihuaijun@cmbchina.com)
 @Version : $Id$
 @Desc:
 pip install tablib py_absl
"""

import time

import pandas as pd
import tablib
from absl import app, logging

from predict2 import predict  # 导入预测方法

MAX_DAY = 30
BEGIN_EVAL = 900
DAY_MAX_TSID = 9000

tick_test_file = "/test/tick_test.csv"            # fake 文件只是为了展示eval的形式，fake的受益与评判无关
label_test_file = "/test/label_test.csv"


def build_tick_dataset(file):
    """
    构建tick数据集
    input:
        file: 可以读的CSV文件
    output:
        dictionary 格式的数据集，day-ts 为key
    """
    dict_format = {}
    dataset = tablib.import_set(open(file, encoding="utf-8"), format="csv")
    header = dataset.headers
    for raw in dataset:
        time_key = "{day}-{ts}".format(day=raw[0], ts=raw[1])
        single_record = dict(zip(header, raw))
        if time_key not in dict_format.keys():
            dict_format[time_key] = [single_record]
        else:
            dict_format[time_key].append(single_record)
    del dataset
    return dict_format


def build_labelset(file):
    """
    构建标签结果集
    input:
        file: 可以读的CSV文件
    output:
        dictionary 格式的数据集，day-ts 为key
    """
    dict_format = {}
    dataset = tablib.import_set(open(file, encoding="utf-8"), format="csv")
    header = dataset.headers
    for raw in dataset:
        time_key = "{day}-{ts}".format(day=raw[0], ts=raw[1])
        single_record = dict(zip(header, raw))
        dict_format[time_key] = single_record
    del dataset
    return dict_format


def label_max(file):
    dataset = tablib.import_set(open(file, encoding="utf-8"), format="csv")
    val = 0.0
    for raw in dataset:
        if int(raw[1]) > BEGIN_EVAL:
            ret = max([float(x) if x else 0 for x in raw[2:]])
            val += ret
    return val


def main(_):
    total_return = 0.0
    logging.info("begin to build tick dataset")
    tick_dataset = build_tick_dataset(tick_test_file)
    logging.info("begin to build label dataset")
    label = build_labelset(label_test_file)
    logging.info("begin to compute the total return")
    start_time = time.time()
    for day in range(MAX_DAY):
        print(day)
        data_test = pd.DataFrame(columns=['day', 'tx_id', 'best_bid_price_1', 'best_bid_size_1','best_ask_price_1', 'best_ask_size_1', 'best_bid_price_2',
                                          'best_bid_size_2', 'best_ask_price_2', 'best_ask_size_2','best_bid_price_3', 'best_bid_size_3', 'best_ask_price_3',
                                          'best_ask_size_3', 'best_bid_price_4', 'best_bid_size_4','best_ask_price_4', 'best_ask_size_4', 'best_bid_price_5',
                                          'best_bid_size_5', 'best_ask_price_5', 'best_ask_size_5', 'last_price','qty'])
        for ts in range(DAY_MAX_TSID):
            # print('!')
            time_key = "{day}-{ts}".format(day=day, ts=ts)
            record = tick_dataset.get(time_key, [])
            for i in range(len(record)):
                the_df = pd.DataFrame(record[i], index=[0])
                data_test = pd.concat([data_test, the_df], axis=0, ignore_index=True)
            #print(ts)
            if len(record) == 0:
                pre_act = 0
                the_ts = the_ts + 1
            else:
                the_ts = int(record[0]['tx_id'])
                
            if the_ts < 900:
                pre_act =1
                
            else:
                pre_act = predict(data_test)[0]+1
            #print(pre_act)
            if ts > BEGIN_EVAL:
                action = "y{d}".format(d=int(pre_act))
                true_return = label.get(time_key, {})
                ret_val = true_return.get(action, None)
                total_return += float(ret_val) if ret_val else 0
    end_time = time.time()
    logging.info("done\n the cost time is {} second".format(end_time - start_time))
    logging.info("the total return is {f}".format(f=total_return))



if __name__ == '__main__':
    ret = label_max(label_test_file)
    print(ret)
    app.run(main)
