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
offline_df = pd.read_csv('../data/ccf_offline_stage1_train.csv')
offline_df = offline_df.fillna(0)
offline_df = offline_df[offline_df['Date_received'] < 20160501]
logger.info(offline_df.shape)

online_df = pd.read_csv('../data/ccf_online_stage1_train.csv')
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
