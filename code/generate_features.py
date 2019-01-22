# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import math

from datetime import datetime
from scipy import stats

import logging

logger = logging.getLogger('ai')
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s')

def join(df, col, series, key):
    return multi_join(df, col, series, [key])

def multi_join(df, col, series, keys):
    t = series.to_frame()
    t.columns = [col]
    return pd.merge(df, t, on=keys, how='left')

def cal_duration(row):
    if row['Coupon_id'] > 0 and row['Date_received'] > 0 and row['Date'] > 0:
        date_received = datetime.strptime(str(int(row['Date_received'])), '%Y%m%d')
        date_consumed = datetime.strptime(str(int(row['Date'])), '%Y%m%d')
        delta = date_consumed - date_received
        return delta.days + 1
    return 0

def cal_previous_duration(row):
    if row['User_id'] == row['Previous_user_id'] and row['Date_received'] > 0 and row['Previous_date_received'] > 0:
        date_received = datetime.strptime(str(int(row['Date_received'])), '%Y%m%d')
        date_previous_received = datetime.strptime(str(int(row['Previous_date_received'])), '%Y%m%d')
        delta = date_received - date_previous_received
        return delta.days + 1
    
    return 0

def cal_next_duration(row):
    if row['User_id'] == row['Next_user_id'] and row['Date_received'] > 0 and row['Next_date_received'] > 0:
        date_received = datetime.strptime(str(int(row['Date_received'])), '%Y%m%d')
        date_next_received = datetime.strptime(str(int(row['Next_date_received'])), '%Y%m%d')
        delta = date_next_received - date_received
        return delta.days + 1
    
    return 0

# 优惠券信息 - 计算折扣率
def cal_discount(row):
    if isinstance(row['Discount_rate'], int):
        return float(row['Discount_rate'])
    
    if row['Discount_rate'] == 'fixed':
        return 0.0
    
    arr = row['Discount_rate'].split(':')
    if len(arr) == 2:
        return (float(arr[0]) - float(arr[1])) / float(arr[0])
    else:
        return float(row['Discount_rate'])

def set_coupon_type(row):
    if isinstance(row['Discount_rate'], int):
        return 1
    
    if row['Discount_rate'] == 'fixed':
        return 2
    
    arr = row['Discount_rate'].split(':')
    if len(arr) == 2:
        return 1
    else:
        return 0

def base_consume(row):
    if isinstance(row['Discount_rate'], int):
        return float(row['Discount_rate'])
    
    if row['Discount_rate'] == 'fixed':
        return 0.0
    
    arr = row['Discount_rate'].split(':')
    if len(arr) == 2:
        return float(arr[0])
    else:
        return 0.0

def get_day_in_month_4_received_day(row):
    if isinstance(row['Date_received'], int) or int(row['Date_received']) <= 0:
        return 0.0
    
    date_received = datetime.strptime(str(int(row['Date_received'])), '%Y%m%d')
    return date_received.day

def get_day_in_week_4_received_day(row):
    if isinstance(row['Date_received'], int) or int(row['Date_received']) <= 0:
        return 0.0
    
    date_received = datetime.strptime(str(int(row['Date_received'])), '%Y%m%d')
    return (date_received.weekday() + 1)

def base_data_process(df):
    df = df.sort_values(by=['User_id', 'Date_received'], ascending=True)

    df['Previous_user_id'] = df['User_id'].shift(1)
    df['Previous_date_received'] = df['Date_received'].shift(1)

    df['Next_user_id'] = df['User_id'].shift(-1)
    df['Next_date_received'] = df['Date_received'].shift(-1)

    df.fillna(0)
    
    df['Distance'] = df['Distance'] + 1
    df['Duration'] = df.apply(lambda row: cal_duration(row), axis=1)
    df['Previous_duration'] = df.apply(lambda row: cal_previous_duration(row), axis=1)
    df['Next_duration'] = df.apply(lambda row: cal_next_duration(row), axis=1)
    
    df = df.drop(['Next_user_id', 'Previous_user_id'], axis=1)
    
    df['Base_consume'] = df.apply(lambda row: base_consume(row), axis=1)
    df['Day_in_month_received'] = df.apply(lambda row: get_day_in_month_4_received_day(row), axis=1)
    df['Day_in_week_received'] = df.apply(lambda row: get_day_in_week_4_received_day(row), axis=1)
    df['Discount'] = df.apply(lambda row: cal_discount(row), axis=1)
    df['Coupon_type'] = df.apply(lambda row: set_coupon_type(row), axis=1)
    
    df['Is_in_day_consume'] = df.apply(lambda row: 1 if row['Duration'] > 0 and row['Duration'] < 17 else 0, axis = 1)
    
    return df

## 加载数据
logger.info('Load source data')
offline_df = pd.read_csv('../source/ccf_offline_stage1_train.csv')
offline_df = offline_df.fillna(0)
offline_df = offline_df[offline_df['Date_received'] < 20160501]
logger.info(offline_df.shape)

online_df = pd.read_csv('../source/ccf_online_stage1_train.csv')
online_df = online_df.fillna(0)
logger.info(online_df.shape)

df = offline_df.copy()
df = base_data_process(df)

## 数据分层
df['Coupon_id'] = df['Coupon_id'].astype('int64', copy=True)

# 领取优惠券的信息
received_df = df[df['Coupon_id'] > 0]

# 消费的信息
consume_df = df[df['Date'] > 0]

# 消费同时15天内使用优惠券的信息
used_coupon_df = df[df['Is_in_day_consume'] == 1]

##################################################################################################################
## 用户特征抽取
# 获取所有在线下消费过的用户id
user_df = df['User_id'].drop_duplicates()
user_df = user_df.to_frame()

# 用户领取优惠券次数
user_receive_count = received_df.groupby(['User_id']).size()
# 用户线下门店消费次数
user_consume_count = consume_df.groupby(['User_id']).size()
# 用户15天内线下门店消费并核销优惠券次数
user_used_count = used_coupon_df.groupby(['User_id']).size()

# 用户特征
user_df = join(user_df, 'User_receive_count', user_receive_count, 'User_id')
user_df = join(user_df, 'User_consume_count', user_consume_count, 'User_id')
user_df = join(user_df, 'User_used_count', user_used_count, 'User_id')

user_df = user_df.fillna(0)
user_df['User_not_used_count'] = user_df.apply(lambda r: r['User_receive_count'] - r['User_used_count'], axis=1)

# 用户15天内线下门店领取优惠券后进行核销率
user_df['User_used_coupon_rate'] = user_df.apply(lambda r: r['User_used_count'] / r['User_receive_count'], axis=1)

user_df = user_df.fillna(0)
user_df = user_df.replace(math.inf, 0)

used_coupon_rate_max = user_df['User_used_coupon_rate'].max()
used_coupon_rate_min = user_df['User_used_coupon_rate'].min()
used_coupon_rate_mean = user_df['User_used_coupon_rate'].mean()

user_df['User_used_coupon_rate_max'] = used_coupon_rate_max
user_df['User_used_coupon_rate_min'] = used_coupon_rate_min
user_df['User_used_coupon_rate_mean'] = used_coupon_rate_mean

user_receive_coupon_merchant_count =  received_df[['User_id','Merchant_id']].drop_duplicates().groupby(['User_id']).size()
user_consume_merchant_count =  consume_df[['User_id','Merchant_id']].drop_duplicates().groupby(['User_id']).size()
user_used_coupon_merchant_count =  used_coupon_df[['User_id','Merchant_id']].drop_duplicates().groupby(['User_id']).size()

user_df = join(user_df, 'User_receive_coupon_merchant_count', user_receive_coupon_merchant_count, 'User_id')
user_df = join(user_df, 'User_consume_merchant_count', user_consume_merchant_count, 'User_id')
user_df = join(user_df, 'User_used_coupon_merchant_count', user_used_coupon_merchant_count, 'User_id')
user_df = user_df.fillna(0)

merchant_count = df['Merchant_id'].drop_duplicates().count()
user_df['User_used_coupon_merchant_occ'] = user_df['User_used_coupon_merchant_count'] / merchant_count

user_df = user_df.fillna(0)
user_df = user_df.replace(math.inf, 0)

user_receive_different_coupon_count =  received_df[['User_id','Coupon_id']].drop_duplicates().groupby(['User_id']).size()
user_used_different_coupon_count =  used_coupon_df[['User_id','Coupon_id']].drop_duplicates().groupby(['User_id']).size()

user_df = join(user_df, 'User_receive_different_coupon_count', user_receive_different_coupon_count, 'User_id')
user_df = join(user_df, 'User_used_different_coupon_count', user_used_different_coupon_count, 'User_id')
user_df = user_df.fillna(0)

coupon_count = df[df['Coupon_id']>0]['Coupon_id'].drop_duplicates().count()

user_df['User_receive_different_coupon_occ'] = user_df['User_receive_different_coupon_count'] / coupon_count
user_df['User_used_different_coupon_occ'] = user_df['User_used_different_coupon_count'] / coupon_count

# 用户平均核销每个商家多少张优惠券
user_df['User_receive_coupon_mean'] = user_df['User_receive_count'] / user_df['User_receive_coupon_merchant_count']
user_df['User_used_coupon_mean'] = user_df['User_used_count'] / user_df['User_receive_coupon_merchant_count']

user_df = user_df.fillna(0)
user_df = user_df.replace(math.inf, 0)

user_distance_used_mean = used_coupon_df[['User_id', 'Distance']].groupby(['User_id']).mean()
user_distance_used_max = used_coupon_df[['User_id', 'Distance']].groupby(['User_id']).max()
user_distance_used_min = used_coupon_df[['User_id', 'Distance']].groupby(['User_id']).min()

user_distance_df = pd.DataFrame({'User_id': user_distance_used_mean.index, 'User_distance_used_mean':user_distance_used_mean['Distance'], 'User_distance_used_max':user_distance_used_max['Distance'], 'User_distance_used_min':user_distance_used_min['Distance']})

user_df = pd.merge(user_df, user_distance_df, on=['User_id'], how='left')
user_df = user_df.fillna(0)

user_duration_used_mean = used_coupon_df[['User_id', 'Duration']].groupby(['User_id']).mean()
user_duration_used_max = used_coupon_df[['User_id', 'Duration']].groupby(['User_id']).max()
user_duration_used_min = used_coupon_df[['User_id', 'Duration']].groupby(['User_id']).min()

user_duration_df = pd.DataFrame({'User_id': user_duration_used_mean.index, 'User_duration_used_mean':user_duration_used_mean['Duration'], 'User_duration_used_max':user_duration_used_max['Duration'], 'User_duration_used_min':user_duration_used_min['Duration']})

user_df = pd.merge(user_df, user_duration_df, on=['User_id'], how='left')
user_df = user_df.fillna(0)

user_previous_duration_used_mean = used_coupon_df[['User_id', 'Previous_duration']].groupby(['User_id']).mean()
user_previous_duration_used_max = used_coupon_df[['User_id', 'Previous_duration']].groupby(['User_id']).max()
user_previous_duration_used_min = used_coupon_df[['User_id', 'Previous_duration']].groupby(['User_id']).min()

user_previous_duration_df = pd.DataFrame({'User_id': user_previous_duration_used_mean.index, 'User_previous_duration_used_mean':user_previous_duration_used_mean['Previous_duration'], 'User_previous_duration_used_max':user_previous_duration_used_max['Previous_duration'], 'User_previous_duration_used_min':user_previous_duration_used_min['Previous_duration']})

user_df = pd.merge(user_df, user_previous_duration_df, on=['User_id'], how='left')
user_df = user_df.fillna(0)

user_next_duration_used_mean = used_coupon_df[['User_id', 'Next_duration']].groupby(['User_id']).mean()
user_next_duration_used_max = used_coupon_df[['User_id', 'Next_duration']].groupby(['User_id']).max()
user_next_duration_used_min = used_coupon_df[['User_id', 'Next_duration']].groupby(['User_id']).min()

user_next_duration_df = pd.DataFrame({'User_id': user_next_duration_used_mean.index, 'User_next_duration_used_mean':user_next_duration_used_mean['Next_duration'], 'User_next_duration_used_max':user_next_duration_used_max['Next_duration'], 'User_next_duration_used_min':user_next_duration_used_min['Next_duration']})

user_df = pd.merge(user_df, user_next_duration_df, on=['User_id'], how='left')
user_df = user_df.fillna(0)

user_df.to_csv('../features/lcm_train_user_features.csv', index=False, header=True)
logger.info('User features extract completely!')

##################################################################################################################
# 获取所有在线下消费过的商户id
merchant_df = df['Merchant_id'].drop_duplicates()
merchant_df = merchant_df.to_frame()

merchant_receive_count = received_df.groupby(['Merchant_id']).size()
merchant_consume_count = consume_df.groupby(['Merchant_id']).size()
merchant_used_count = used_coupon_df.groupby(['Merchant_id']).size()

# 商户特征
merchant_df = join(merchant_df, 'Merchant_receive_count', merchant_receive_count, 'Merchant_id')
merchant_df = join(merchant_df, 'Merchant_consume_count', merchant_consume_count, 'Merchant_id')
merchant_df = join(merchant_df, 'Merchant_used_count', merchant_used_count, 'Merchant_id')
merchant_df = merchant_df.fillna(0)

merchant_df['Merchant_not_used_count'] = merchant_df.apply(lambda r: r['Merchant_receive_count'] - r['Merchant_used_count'], axis=1)

# 商户15天内线下门店领取优惠券后进行核销率
merchant_df['Merchant_used_coupon_rate'] = merchant_df.apply(lambda r: r['Merchant_used_count'] / r['Merchant_receive_count'], axis=1)

merchant_df = merchant_df.fillna(0)
merchant_df = merchant_df.replace(math.inf, 0)

merchant_df['Merchant_used_coupon_rate_max'] = merchant_df['Merchant_used_coupon_rate'].max()
merchant_df['Merchant_used_coupon_rate_min'] = merchant_df['Merchant_used_coupon_rate'].min()
merchant_df['Merchant_used_coupon_rate_mean'] = merchant_df['Merchant_used_coupon_rate'].mean()

merchant_receive_coupon_user_count =  received_df[['User_id','Merchant_id']].drop_duplicates().groupby(['Merchant_id']).size()
merchant_consume_user_count =  consume_df[['User_id','Merchant_id']].drop_duplicates().groupby(['Merchant_id']).size()
merchant_used_coupon_user_count =  used_coupon_df[['User_id','Merchant_id']].drop_duplicates().groupby(['Merchant_id']).size()

merchant_df = join(merchant_df, 'Merchant_receive_coupon_user_count', merchant_receive_coupon_user_count, 'Merchant_id')
merchant_df = join(merchant_df, 'Merchant_consume_user_count', merchant_consume_user_count, 'Merchant_id')
merchant_df = join(merchant_df, 'Merchant_used_coupon_user_count', merchant_used_coupon_user_count, 'Merchant_id')
merchant_df = merchant_df.fillna(0)

user_count = df['User_id'].drop_duplicates().count()

merchant_df['Merchant_receive_coupon_user_occ'] = merchant_df['Merchant_receive_coupon_user_count'] / user_count
merchant_df['Merchant_consume_user_occ'] = merchant_df['Merchant_consume_user_count'] / user_count
merchant_df['Merchant_used_coupon_user_occ'] = merchant_df['Merchant_used_coupon_user_count'] / user_count

merchant_receive_different_coupon_count =  received_df[['Merchant_id','Coupon_id']].drop_duplicates().groupby(['Merchant_id']).size()
merchant_used_different_coupon_count =  used_coupon_df[['Merchant_id','Coupon_id']].drop_duplicates().groupby(['Merchant_id']).size()

merchant_df = join(merchant_df, 'Merchant_receive_different_coupon_count', merchant_receive_different_coupon_count, 'Merchant_id')
merchant_df = join(merchant_df, 'Merchant_used_different_coupon_count', merchant_used_different_coupon_count, 'Merchant_id')
merchant_df = merchant_df.fillna(0)

merchant_df['Merchant_receive_different_coupon_occ'] = merchant_df['Merchant_receive_different_coupon_count'] / coupon_count
merchant_df['Merchant_used_different_coupon_occ'] = merchant_df['Merchant_used_different_coupon_count'] / coupon_count

merchant_df['Merchant_receive_coupon_mean'] = merchant_df['Merchant_receive_count'] / merchant_df['Merchant_receive_coupon_user_count']
merchant_df['Merchant_used_coupon_mean'] = merchant_df['Merchant_used_count'] / merchant_df['Merchant_used_coupon_user_count']

merchant_df = merchant_df.replace(math.inf, 0)
merchant_df = merchant_df.fillna(0)

merchant_df['Merchant_receive_different_coupon_avg'] = merchant_df['Merchant_receive_count'] / merchant_df['Merchant_receive_different_coupon_count']
merchant_df['Merchant_used_different_coupon_avg'] = merchant_df['Merchant_receive_count'] / merchant_df['Merchant_used_coupon_user_count']

merchant_df = merchant_df.replace(math.inf, 0)
merchant_df = merchant_df.fillna(0)

merchant_distance_used_mean = used_coupon_df[['Merchant_id', 'Distance']].groupby(['Merchant_id']).mean()
merchant_distance_used_max = used_coupon_df[['Merchant_id', 'Distance']].groupby(['Merchant_id']).max()
merchant_distance_used_min = used_coupon_df[['Merchant_id', 'Distance']].groupby(['Merchant_id']).min()

merchant_distance_df = pd.DataFrame({'Merchant_id': merchant_distance_used_mean.index, 'Merchant_distance_used_mean':merchant_distance_used_mean['Distance'], 'Merchant_distance_used_max':merchant_distance_used_max['Distance'], 'Merchant_distance_used_min':merchant_distance_used_min['Distance']})

merchant_df = pd.merge(merchant_df, merchant_distance_df, on=['Merchant_id'], how='left')
merchant_df = merchant_df.fillna(0)

merchant_duration_used_mean = used_coupon_df[['Merchant_id', 'Duration']].groupby(['Merchant_id']).mean()
merchant_duration_used_max = used_coupon_df[['Merchant_id', 'Duration']].groupby(['Merchant_id']).max()
merchant_duration_used_min = used_coupon_df[['Merchant_id', 'Duration']].groupby(['Merchant_id']).min()

merchant_duration_df = pd.DataFrame({'Merchant_id': merchant_duration_used_mean.index, 'Merchant_duration_used_mean':merchant_duration_used_mean['Duration'], 'Merchant_duration_used_max':merchant_duration_used_max['Duration'], 'Merchant_duration_used_min':merchant_duration_used_min['Duration']})

merchant_df = pd.merge(merchant_df, merchant_duration_df, on=['Merchant_id'], how='left')
merchant_df = merchant_df.fillna(0)

merchant_previous_duration_used_mean = used_coupon_df[['Merchant_id', 'Previous_duration']].groupby(['Merchant_id']).mean()
merchant_previous_duration_used_max = used_coupon_df[['Merchant_id', 'Previous_duration']].groupby(['Merchant_id']).max()
merchant_previous_duration_used_min = used_coupon_df[['Merchant_id', 'Previous_duration']].groupby(['Merchant_id']).min()

merchant_previous_duration_df = pd.DataFrame({'Merchant_id': merchant_previous_duration_used_mean.index, 'Merchant_previous_duration_used_mean':merchant_previous_duration_used_mean['Previous_duration'], 'Merchant_previous_duration_used_max':merchant_previous_duration_used_max['Previous_duration'], 'Merchant_previous_duration_used_min':merchant_previous_duration_used_min['Previous_duration']})

merchant_df = pd.merge(merchant_df, merchant_previous_duration_df, on=['Merchant_id'], how='left')
merchant_df = merchant_df.fillna(0)

merchant_next_duration_used_mean = used_coupon_df[['Merchant_id', 'Next_duration']].groupby(['Merchant_id']).mean()
merchant_next_duration_used_max = used_coupon_df[['Merchant_id', 'Next_duration']].groupby(['Merchant_id']).max()
merchant_next_duration_used_min = used_coupon_df[['Merchant_id', 'Next_duration']].groupby(['Merchant_id']).min()

merchant_next_duration_df = pd.DataFrame({'Merchant_id': merchant_next_duration_used_mean.index, 'Merchant_next_duration_used_mean':merchant_next_duration_used_mean['Next_duration'], 'Merchant_next_duration_used_max':merchant_next_duration_used_max['Next_duration'], 'Merchant_next_duration_used_min':merchant_next_duration_used_min['Next_duration']})

merchant_df = pd.merge(merchant_df, merchant_next_duration_df, on=['Merchant_id'], how='left')
merchant_df = merchant_df.fillna(0)

merchant_df.to_csv('../features/lcm_train_merchant_features.csv', index=False, header=True)
logger.info('Merchant features extract completely!')

##################################################################################################################
# 获取所有在线下消费过的优惠券id
coupon_df = df[df['Coupon_id']>0]['Coupon_id'].drop_duplicates()
coupon_df = coupon_df.to_frame()

coupon_received_count = received_df.groupby(['Coupon_id']).size()
coupon_used_count = used_coupon_df.groupby(['Coupon_id']).size()

coupon_df = join(coupon_df, 'Coupon_received_count', coupon_received_count, 'Coupon_id')
coupon_df = join(coupon_df, 'Coupon_used_count', coupon_used_count, 'Coupon_id')
coupon_df = coupon_df.fillna(0)

coupon_df['Coupon_used_rate'] = coupon_df['Coupon_used_count'] / coupon_df['Coupon_received_count']

coupon_df = coupon_df.replace(math.inf, 0)
coupon_df = coupon_df.fillna(0)

coupon_duration_used_mean = used_coupon_df[['Coupon_id', 'Duration']].groupby(['Coupon_id']).mean()
coupon_duration_used_max = used_coupon_df[['Coupon_id', 'Duration']].groupby(['Coupon_id']).max()
coupon_duration_used_min = used_coupon_df[['Coupon_id', 'Duration']].groupby(['Coupon_id']).min()

coupon_duration_df = pd.DataFrame({'Coupon_id': coupon_duration_used_mean.index, 'Coupon_duration_used_mean':coupon_duration_used_mean['Duration'], 'Coupon_duration_used_max':coupon_duration_used_max['Duration'], 'Coupon_duration_used_min':coupon_duration_used_min['Duration']})

coupon_df = pd.merge(coupon_df, coupon_duration_df, on=['Coupon_id'], how='left')
coupon_df = coupon_df.fillna(0)

coupon_distance_used_mean = used_coupon_df[['Coupon_id', 'Distance']].groupby(['Coupon_id']).mean()
coupon_distance_used_max = used_coupon_df[['Coupon_id', 'Distance']].groupby(['Coupon_id']).max()
coupon_distance_used_min = used_coupon_df[['Coupon_id', 'Distance']].groupby(['Coupon_id']).min()

coupon_distance_df = pd.DataFrame({'Coupon_id': coupon_distance_used_mean.index, 'Coupon_distance_used_mean':coupon_distance_used_mean['Distance'], 'Coupon_distance_used_max':coupon_distance_used_max['Distance'], 'Coupon_distance_used_min':coupon_distance_used_min['Distance']})

coupon_df = pd.merge(coupon_df, coupon_distance_df, on=['Coupon_id'], how='left')
coupon_df = coupon_df.fillna(0)

coupon_df.to_csv('../features/lcm_train_coupon_features.csv', index=False, header=True)
logger.info('Coupon features extract completely!')

##################################################################################################################
# 获取所有在线下消费过的商户id
user_merchant_df = df[['User_id','Merchant_id']].drop_duplicates()

user_merchant_receive_count = received_df.groupby(['User_id', 'Merchant_id']).size()
user_merchant_consume_count = consume_df.groupby(['User_id', 'Merchant_id']).size()
user_merchant_used_count = used_coupon_df.groupby(['User_id', 'Merchant_id']).size()

user_merchant_df = multi_join(user_merchant_df, 'User_merchant_receive_count', user_merchant_receive_count, ['Merchant_id', 'User_id'])
user_merchant_df = multi_join(user_merchant_df, 'User_merchant_consume_count', user_merchant_consume_count, ['Merchant_id', 'User_id'])
user_merchant_df = multi_join(user_merchant_df, 'User_merchant_used_count', user_merchant_used_count, ['Merchant_id', 'User_id'])
user_merchant_df = user_merchant_df.fillna(0)

user_merchant_df['User_merchant_not_used_count'] = user_merchant_df.apply(lambda r: r['User_merchant_receive_count'] - r['User_merchant_used_count'], axis=1)

# 用户领取优惠券次数
user_receive_count = received_df.groupby(['User_id']).size()
# 用户15天内线下门店消费并核销优惠券次数
user_used_count = used_coupon_df.groupby(['User_id']).size()

user_merchant_df = join(user_merchant_df, 'User_receive_count', user_receive_count, 'User_id')
user_merchant_df = join(user_merchant_df, 'User_used_count', user_used_count, 'User_id')

user_merchant_df['User_merchant_used_coupon_rate'] = user_merchant_df['User_merchant_used_count'] / user_merchant_df['User_used_count']
user_merchant_df['User_merchant_not_used_coupon_rate'] = user_merchant_df['User_merchant_not_used_count'] / user_merchant_df['User_used_count']

user_merchant_df = user_merchant_df.fillna(0)
user_merchant_df = user_merchant_df.replace(math.inf, 0)

merchant_receive_count = received_df.groupby(['Merchant_id']).size()
merchant_used_count = used_coupon_df.groupby(['Merchant_id']).size()

user_merchant_df = join(user_merchant_df, 'Merchant_receive_count', merchant_receive_count, 'Merchant_id')
user_merchant_df = join(user_merchant_df, 'Merchant_used_count', merchant_used_count, 'Merchant_id')

user_merchant_df['User_merchant_used_coupon_rate_4_merchant'] = user_merchant_df['User_merchant_used_count'] / user_merchant_df['Merchant_used_count']
user_merchant_df['User_merchant_not_used_coupon_rate_4_merchant'] = user_merchant_df['User_merchant_not_used_count'] / user_merchant_df['Merchant_used_count']

user_merchant_df = user_merchant_df.fillna(0)
user_merchant_df = user_merchant_df.replace(math.inf, 0)

user_merchant_duration_used_mean = used_coupon_df[['User_id', 'Merchant_id', 'Duration']].groupby(['User_id', 'Merchant_id']).mean()
user_merchant_duration_used_max = used_coupon_df[['User_id', 'Merchant_id', 'Duration']].groupby(['User_id', 'Merchant_id']).max()
user_merchant_duration_used_min = used_coupon_df[['User_id', 'Merchant_id',  'Duration']].groupby(['User_id', 'Merchant_id']).min()

user_merchant_duration_used_mean = user_merchant_duration_used_mean.rename(columns={'Duration':'User_merchant_duration_used_mean'})
user_merchant_df = pd.merge(user_merchant_df, user_merchant_duration_used_mean, on=['Merchant_id', 'User_id'], how='left')

user_merchant_duration_used_max = user_merchant_duration_used_max.rename(columns={'Duration':'User_merchant_duration_used_max'})
user_merchant_df = pd.merge(user_merchant_df, user_merchant_duration_used_max, on=['Merchant_id', 'User_id'], how='left')

user_merchant_duration_used_min = user_merchant_duration_used_min.rename(columns={'Duration':'User_merchant_duration_used_min'})
user_merchant_df = pd.merge(user_merchant_df, user_merchant_duration_used_min, on=['Merchant_id', 'User_id'], how='left')

user_merchant_df = user_merchant_df.fillna(0)

user_merchant_df.to_csv('../features/lcm_train_user_merchant_features.csv', index=False, header=True)
logger.info('User merchant features extract completely!')

##################################################################################################################
def check_is_in_day_consume(row):
    
    if row['Coupon_id'] == 'fixed':
        return 0
    
    if float(row['Coupon_id']) > 0 and float(row['Date_received']) > 0 and float(row['Date']) > 0:
        date_received = datetime.strptime(str(int(row['Date_received'])), '%Y%m%d')
        date_consumed = datetime.strptime(str(int(row['Date'])), '%Y%m%d')
        delta = date_consumed - date_received
        if delta.days < 16:
            return 1
        else:
            return 0
    
    return 0

online_df['Is_in_day_consume'] = online_df.apply(lambda row: check_is_in_day_consume(row), axis=1)

online_df['Coupon_id'] = online_df['Coupon_id'].replace('fixed', 0)
online_df['Coupon_id'] = online_df['Coupon_id'].astype('int64', copy=True)

# 领取优惠券的信息
online_received_df = online_df[online_df['Coupon_id'] > 0]

# 消费的信息
online_consume_df = online_df[online_df['Date'] > 0]

# 消费同时15天内使用优惠券的信息
online_used_coupon_df = online_df[online_df['Is_in_day_consume'] == 1]

# 获取所有在线下消费过的用户id
user_online_df = online_df['User_id'].drop_duplicates()
user_online_df = user_online_df.to_frame()

# 用户领取优惠券次数
online_user_receive_count = online_received_df.groupby(['User_id']).size()
# 用户线上门店消费次数
online_user_consume_count = online_consume_df.groupby(['User_id']).size()
# 用户15天内线下门店消费并核销优惠券次数
online_user_used_count = online_used_coupon_df.groupby(['User_id']).size()

# 用户特征
user_online_df = join(user_online_df, 'Online_user_receive_count', online_user_receive_count, 'User_id')
user_online_df = join(user_online_df, 'Online_user_consume_count', online_user_consume_count, 'User_id')
user_online_df = join(user_online_df, 'Online_user_used_count', online_user_used_count, 'User_id')
user_online_df = user_online_df.fillna(0)
user_online_df['Online_user_not_used_count'] = user_online_df.apply(lambda r: r['Online_user_receive_count'] - r['Online_user_used_count'], axis=1)

# 用户15天内线下门店领取优惠券后进行核销率
user_online_df['Online_user_used_coupon_rate'] = user_online_df.apply(lambda r: r['Online_user_used_count'] / r['Online_user_receive_count'], axis=1)
user_online_df = user_online_df.fillna(0)
user_online_df = user_online_df.replace(math.inf, 0)

user_online_df = join(user_online_df, 'User_receive_count', user_receive_count, 'User_id')
user_online_df = join(user_online_df, 'User_consume_count', user_consume_count, 'User_id')
user_online_df = join(user_online_df, 'User_used_count', user_used_count, 'User_id')
user_online_df['User_not_used_count'] = user_online_df.apply(lambda r: r['User_receive_count'] - r['User_used_count'], axis=1)
user_online_df = user_online_df.fillna(0)

user_online_df['User_offline_consume_rate'] = user_online_df.apply(lambda r: r['User_consume_count'] / (r['Online_user_consume_count'] + r['User_consume_count']), axis=1)  
user_online_df['User_offline_used_rate'] = user_online_df.apply(lambda r: r['User_used_count'] / (r['Online_user_used_count'] + r['User_used_count']), axis=1)
user_online_df['User_offline_no_consume_coupon_rate'] = user_online_df.apply(lambda r: r['User_not_used_count'] / (r['Online_user_not_used_count'] + r['User_not_used_count']), axis=1) 

user_online_df = user_online_df.fillna(0)
user_online_df = user_online_df.replace(math.inf, 0)

user_online_df.to_csv('../features/lcm_train_user_online_features.csv', index=False, header=True)
logger.info('User online features extract completely!')

##################################################################################################################
df = pd.merge(df, user_df, on=['User_id'], how='left')
df = pd.merge(df, merchant_df, on=['Merchant_id'], how='left')
df = pd.merge(df, coupon_df, on=['Coupon_id'], how='left')

user_merchant_df = user_merchant_df.drop(['User_receive_count', 'User_used_count', 'Merchant_receive_count', 'Merchant_used_count'], axis=1)

df = pd.merge(df, user_merchant_df, on=['User_id', 'Merchant_id'], how='left')

user_online_df = user_online_df.drop(['User_receive_count', 'User_consume_count', 'User_used_count', 'User_not_used_count'], axis=1)

df = pd.merge(df, user_online_df, on=['User_id'], how='left')
df = df.fillna(0)

user_distance_receive_count = received_df.groupby(['User_id', 'Distance']).size()
user_distance_consume_count = consume_df.groupby(['User_id', 'Distance']).size()
user_distance_used_count = used_coupon_df.groupby(['User_id', 'Distance']).size()

# 用户-距离特征
df = multi_join(df, 'User_distance_receive_count', user_distance_receive_count, ['User_id', 'Distance'])
df = multi_join(df, 'User_distance_consume_count', user_distance_consume_count, ['User_id', 'Distance'])
df = multi_join(df, 'User_distance_used_count', user_distance_used_count, ['User_id', 'Distance'])
df = df.fillna(0)

df['User_distance_receive_rate'] = df['User_distance_receive_count'] / df['User_receive_count']
df['User_distance_consume_rate'] = df['User_distance_consume_count'] / df['User_consume_count']
df['User_distance_used_rate'] = df['User_distance_used_count'] / df['User_receive_count']
df = df.fillna(0)

user_coupon_type_receive_count = received_df.groupby(['User_id', 'Coupon_type']).size()
user_coupon_type_used_count = used_coupon_df.groupby(['User_id', 'Coupon_type']).size()

# 用户-优惠券类型特征
df = multi_join(df, 'User_coupon_type_receive_count', user_coupon_type_receive_count, ['User_id', 'Coupon_type'])
df = multi_join(df, 'User_coupon_type_used_count', user_coupon_type_used_count, ['User_id', 'Coupon_type'])
df = df.fillna(0)

df['User_coupon_type_receive_rate'] = df['User_coupon_type_receive_count'] / df['User_receive_count']
df['User_coupon_type_used_rate'] = df['User_coupon_type_used_count'] / df['User_receive_count']

user_coupon_receive_count = received_df.groupby(['User_id', 'Coupon_id']).size()
user_coupon_used_count = used_coupon_df.groupby(['User_id', 'Coupon_id']).size()

# 用户-优惠券特征
df = multi_join(df, 'User_coupon_receive_count', user_coupon_receive_count, ['User_id', 'Coupon_id'])
df = multi_join(df, 'User_coupon_used_count', user_coupon_used_count, ['User_id', 'Coupon_id'])
df = df.fillna(0)

df['User_coupon_receive_rate'] = df['User_coupon_receive_count'] / df['User_receive_count']
df['User_coupon_used_rate'] = df['User_coupon_used_count'] / df['User_receive_count']
df = df.fillna(0)

merchant_distance_receive_count = received_df.groupby(['Merchant_id', 'Distance']).size()
merchant_distance_consume_count = consume_df.groupby(['Merchant_id', 'Distance']).size()
merchant_distance_used_count = used_coupon_df.groupby(['Merchant_id', 'Distance']).size()

# 商户-距离特征
df = multi_join(df, 'Merchant_distance_receive_count', merchant_distance_receive_count, ['Distance', 'Merchant_id'])
df = multi_join(df, 'Merchant_distance_consume_count', merchant_distance_consume_count, ['Distance', 'Merchant_id'])
df = multi_join(df, 'Merchant_distance_used_count', merchant_distance_used_count, ['Distance', 'Merchant_id'])
df = df.fillna(0)

df['Merchant_distance_receive_rate'] = df['Merchant_distance_receive_count'] / df['Merchant_receive_count']
df['Merchant_distance_used_rate'] = df['Merchant_distance_used_count'] / df['Merchant_receive_count']
df = df.fillna(0)

user_coupon_duration_used_mean = used_coupon_df[['User_id', 'Coupon_id', 'Duration']].groupby(['User_id', 'Coupon_id']).mean()
user_coupon_duration_used_max = used_coupon_df[['User_id', 'Coupon_id', 'Duration']].groupby(['User_id', 'Coupon_id']).max()
user_coupon_duration_used_min = used_coupon_df[['User_id', 'Coupon_id', 'Duration']].groupby(['User_id', 'Coupon_id']).min()

user_coupon_duration_used_mean = user_coupon_duration_used_mean.rename(columns={'Duration':'User_coupon_duration_used_mean'})
df = pd.merge(df, user_coupon_duration_used_mean, on=['Coupon_id', 'User_id'], how='left')

user_coupon_duration_used_max = user_coupon_duration_used_max.rename(columns={'Duration':'User_coupon_duration_used_max'})
df = pd.merge(df, user_coupon_duration_used_max, on=['Coupon_id', 'User_id'], how='left')

user_coupon_duration_used_min = user_coupon_duration_used_min.rename(columns={'Duration':'User_coupon_duration_used_min'})
df = pd.merge(df, user_coupon_duration_used_min, on=['Coupon_id', 'User_id'], how='left')

user_received_date_count = df[['User_id', 'Date_received']].groupby(['User_id']).size()
df = multi_join(df, 'User_received_date_count', user_received_date_count, ['User_id'])

df = df.replace(math.inf, 0)
df = df.fillna(0)

df.to_csv('../features/lcm_train_features.csv', index=False, header=True)
logger.info('Train features extract completely!')

##################################################################################################################
df_train_test = pd.read_csv('../source/ccf_offline_stage1_train.csv')
df_train_test = df_train_test[df_train_test['Date_received']>=20160501]
df_train_test = df_train_test.fillna(0)
df_train_test = base_data_process(df_train_test)

df_train_test = pd.merge(df_train_test, user_df, on=['User_id'], how='left')
df_train_test = pd.merge(df_train_test, merchant_df, on=['Merchant_id'], how='left')
df_train_test = pd.merge(df_train_test, coupon_df, on=['Coupon_id'], how='left')
df_train_test = pd.merge(df_train_test, user_merchant_df, on=['User_id', 'Merchant_id'], how='left')
df_train_test = pd.merge(df_train_test, user_online_df, on=['User_id'], how='left')

# 用户-距离特征
df_train_test = multi_join(df_train_test, 'User_distance_receive_count', user_distance_receive_count, ['User_id', 'Distance'])
df_train_test = multi_join(df_train_test, 'User_distance_consume_count', user_distance_consume_count, ['User_id', 'Distance'])
df_train_test = multi_join(df_train_test, 'User_distance_used_count', user_distance_used_count, ['User_id', 'Distance'])
df_train_test = df_train_test.fillna(0)

df_train_test['User_distance_receive_rate'] = df_train_test['User_distance_receive_count'] / df_train_test['User_receive_count']
df_train_test['User_distance_consume_rate'] = df_train_test['User_distance_consume_count'] / df_train_test['User_consume_count']
df_train_test['User_distance_used_rate'] = df_train_test['User_distance_used_count'] / df_train_test['User_receive_count']
df_train_test = df_train_test.fillna(0)

# 用户-优惠券类型特征
df_train_test = multi_join(df_train_test, 'User_coupon_type_receive_count', user_coupon_type_receive_count, ['User_id', 'Coupon_type'])
df_train_test = multi_join(df_train_test, 'User_coupon_type_used_count', user_coupon_type_used_count, ['User_id', 'Coupon_type'])
df_train_test = df_train_test.fillna(0)

df_train_test['User_coupon_type_receive_rate'] = df_train_test['User_coupon_type_receive_count'] / df_train_test['User_receive_count']
df_train_test['User_coupon_type_used_rate'] = df_train_test['User_coupon_type_used_count'] / df_train_test['User_receive_count']

# 用户-优惠券特征
df_train_test = multi_join(df_train_test, 'User_coupon_receive_count', user_coupon_receive_count, ['User_id', 'Coupon_id'])
df_train_test = multi_join(df_train_test, 'User_coupon_used_count', user_coupon_used_count, ['User_id', 'Coupon_id'])
df_train_test = df_train_test.fillna(0)

df_train_test['User_coupon_receive_rate'] = df_train_test['User_coupon_receive_count'] / df_train_test['User_receive_count']
df_train_test['User_coupon_used_rate'] = df_train_test['User_coupon_used_count'] / df_train_test['User_receive_count']
df_train_test = df_train_test.fillna(0)

# 商户-距离特征
df_train_test = multi_join(df_train_test, 'Merchant_distance_receive_count', merchant_distance_receive_count, ['Distance', 'Merchant_id'])
df_train_test = multi_join(df_train_test, 'Merchant_distance_consume_count', merchant_distance_consume_count, ['Distance', 'Merchant_id'])
df_train_test = multi_join(df_train_test, 'Merchant_distance_used_count', merchant_distance_used_count, ['Distance', 'Merchant_id'])
df_train_test = df_train_test.fillna(0)

df_train_test['Merchant_distance_receive_rate'] = df_train_test['Merchant_distance_receive_count'] / df_train_test['Merchant_receive_count']
df_train_test['Merchant_distance_used_rate'] = df_train_test['Merchant_distance_used_count'] / df_train_test['Merchant_receive_count']
df_train_test = df_train_test.fillna(0)

user_coupon_duration_used_mean = user_coupon_duration_used_mean.rename(columns={'Duration':'User_coupon_duration_used_mean'})
df_train_test = pd.merge(df_train_test, user_coupon_duration_used_mean, on=['Coupon_id', 'User_id'], how='left')

user_coupon_duration_used_max = user_coupon_duration_used_max.rename(columns={'Duration':'User_coupon_duration_used_max'})
df_train_test = pd.merge(df_train_test, user_coupon_duration_used_max, on=['Coupon_id', 'User_id'], how='left')

user_coupon_duration_used_min = user_coupon_duration_used_min.rename(columns={'Duration':'User_coupon_duration_used_min'})
df_train_test = pd.merge(df_train_test, user_coupon_duration_used_min, on=['Coupon_id', 'User_id'], how='left')

user_received_date_count = df_train_test[['User_id', 'Date_received']].groupby(['User_id']).size()
df_train_test = multi_join(df_train_test, 'User_received_date_count', user_received_date_count, ['User_id'])

df_train_test = df_train_test.replace(math.inf, 0)
df_train_test = df_train_test.fillna(0)

df_train_test.to_csv('../features/lcm_train_test_features.csv', index=False, header=True)
logger.info('Valid features extract completely!')

##################################################################################################################
test_df = pd.read_csv('../source/ccf_offline_stage1_test_revised.csv')
test_df = test_df.fillna(0)
def base_predict_data_process(df):
    df = df.sort_values(by=['User_id', 'Date_received'], ascending=True)

    df['Previous_user_id'] = df['User_id'].shift(1)
    df['Previous_date_received'] = df['Date_received'].shift(1)

    df['Next_user_id'] = df['User_id'].shift(-1)
    df['Next_date_received'] = df['Date_received'].shift(-1)

    df.fillna(0)
    
    df['Distance'] = df['Distance'] + 1
    df['Previous_duration'] = df.apply(lambda row: cal_previous_duration(row), axis=1)
    df['Next_duration'] = df.apply(lambda row: cal_next_duration(row), axis=1)
    
    df = df.drop(['Next_user_id', 'Previous_user_id'], axis=1)
    
    df['Base_consume'] = df.apply(lambda row: base_consume(row), axis=1)
    df['Day_in_month_received'] = df.apply(lambda row: get_day_in_month_4_received_day(row), axis=1)
    df['Day_in_week_received'] = df.apply(lambda row: get_day_in_week_4_received_day(row), axis=1)
    df['Discount'] = df.apply(lambda row: cal_discount(row), axis=1)
    df['Coupon_type'] = df.apply(lambda row: set_coupon_type(row), axis=1)
    
    return df

test_df = base_predict_data_process(test_df)
test_df = pd.merge(test_df, user_df, on=['User_id'], how='left')
test_df = pd.merge(test_df, merchant_df, on=['Merchant_id'], how='left')
test_df = pd.merge(test_df, coupon_df, on=['Coupon_id'], how='left')
test_df = pd.merge(test_df, user_merchant_df, on=['User_id', 'Merchant_id'], how='left')
test_df = pd.merge(test_df, user_online_df, on=['User_id'], how='left')

# 用户-距离特征
test_df = multi_join(test_df, 'User_distance_receive_count', user_distance_receive_count, ['User_id', 'Distance'])
test_df = multi_join(test_df, 'User_distance_consume_count', user_distance_consume_count, ['User_id', 'Distance'])
test_df = multi_join(test_df, 'User_distance_used_count', user_distance_used_count, ['User_id', 'Distance'])
test_df = test_df.fillna(0)

test_df['User_distance_receive_rate'] = test_df['User_distance_receive_count'] / test_df['User_receive_count']
test_df['User_distance_consume_rate'] = test_df['User_distance_consume_count'] / test_df['User_consume_count']
test_df['User_distance_used_rate'] = test_df['User_distance_used_count'] / test_df['User_receive_count']
test_df = test_df.fillna(0)

# 用户-优惠券类型特征
test_df = multi_join(test_df, 'User_coupon_type_receive_count', user_coupon_type_receive_count, ['User_id', 'Coupon_type'])
test_df = multi_join(test_df, 'User_coupon_type_used_count', user_coupon_type_used_count, ['User_id', 'Coupon_type'])
test_df = test_df.fillna(0)

test_df['User_coupon_type_receive_rate'] = test_df['User_coupon_type_receive_count'] / test_df['User_receive_count']
test_df['User_coupon_type_used_rate'] = test_df['User_coupon_type_used_count'] / test_df['User_receive_count']

# 用户-优惠券特征
test_df = multi_join(test_df, 'User_coupon_receive_count', user_coupon_receive_count, ['User_id', 'Coupon_id'])
test_df = multi_join(test_df, 'User_coupon_used_count', user_coupon_used_count, ['User_id', 'Coupon_id'])
test_df = test_df.fillna(0)

test_df['User_coupon_receive_rate'] = test_df['User_coupon_receive_count'] / test_df['User_receive_count']
test_df['User_coupon_used_rate'] = test_df['User_coupon_used_count'] / test_df['User_receive_count']
test_df = test_df.fillna(0)

# 商户-距离特征
test_df = multi_join(test_df, 'Merchant_distance_receive_count', merchant_distance_receive_count, ['Distance', 'Merchant_id'])
test_df = multi_join(test_df, 'Merchant_distance_consume_count', merchant_distance_consume_count, ['Distance', 'Merchant_id'])
test_df = multi_join(test_df, 'Merchant_distance_used_count', merchant_distance_used_count, ['Distance', 'Merchant_id'])
test_df = test_df.fillna(0)

test_df['Merchant_distance_receive_rate'] = test_df['Merchant_distance_receive_count'] / test_df['Merchant_receive_count']
test_df['Merchant_distance_used_rate'] = test_df['Merchant_distance_used_count'] / test_df['Merchant_receive_count']
test_df = test_df.fillna(0)

user_coupon_duration_used_mean = user_coupon_duration_used_mean.rename(columns={'Duration':'User_coupon_duration_used_mean'})
test_df = pd.merge(test_df, user_coupon_duration_used_mean, on=['Coupon_id', 'User_id'], how='left')

user_coupon_duration_used_max = user_coupon_duration_used_max.rename(columns={'Duration':'User_coupon_duration_used_max'})
test_df = pd.merge(test_df, user_coupon_duration_used_max, on=['Coupon_id', 'User_id'], how='left')

user_coupon_duration_used_min = user_coupon_duration_used_min.rename(columns={'Duration':'User_coupon_duration_used_min'})
test_df = pd.merge(test_df, user_coupon_duration_used_min, on=['Coupon_id', 'User_id'], how='left')

user_received_date_count = test_df[['User_id', 'Date_received']].groupby(['User_id']).size()
test_df = multi_join(test_df, 'User_received_date_count', user_received_date_count, ['User_id'])

test_df = test_df.replace(math.inf, 0)
test_df = test_df.fillna(0)

test_df.to_csv('../features/lcm_test_features.csv', index=False, header=True)
logger.info('Test features extract completely!')


