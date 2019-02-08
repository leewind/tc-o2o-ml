# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import math
import logging
from datetime import datetime

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, Normalizer, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_curve, auc

from mlxtend.preprocessing import DenseTransformer
from mlxtend.feature_selection import ColumnSelector

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

logger = logging.getLogger('ai')
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s')

dataset_alpha = pd.read_csv('../features/dataset_alpha.csv')
dataset_beta = pd.read_csv('../features/dataset_beta.csv')
dataset_pred = pd.read_csv('../features/dataset_pred.csv')

dataset_beta = pd.concat([dataset_alpha, dataset_beta])
dataset_alpha = dataset_pred

continous = [
    'Coupon_id', 'Distance',
    'Month_of_received', 'Day_of_received',
    'Weekday_of_received', 'Base_consume', 'Discount',
    'Discount_money', 'Coupon_type', 'Coupon_category',
    'Previous_duration', 'Next_duration', 'o1',
    'o2', 'o3', 'o4', 'o5', 'o6', 'o8', 'o7', 'o9', 'o10', 'o12',
    'o14', 'o11', 'o13', 'o16', 'o15', 'o18', 'o19', 'o20', 'o21',
    'o22', 'o23', 'o17', 'o24', 'o25', 'o26', 'o27', 'o28', 'o29',
    'o30', 'o38', 'o31', 'o39', 'o40', 'o41', 'o42', 'o43', 'o32',
    'o33', 'o34', 'o35', 'o36', 'o37', 'o44', 'u0', 'u1', 'u2', 'u3',
    'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u12', 'u13',
    'u14', 'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u21', 'u22',
    'u23', 'u24', 'u25', 'ucc0', 'ucc1', 'ucc2', 'ucc3', 'ucc4',
    'ucc5', 'ucc6', 'ucc7', 'ucc8', 'ucc9', 'ucc10', 'ucc11', 'ucc12',
    'uc1', 'uc2', 'uc3', 'uc4', 'uc5', 'uc6', 'uc7', 'uc8', 'uc9',
    'uc10', 'uc11', 'uc12', 'ud0', 'ud1', 'ud2', 'ud3', 'ud4', 'ud5',
    'ud6', 'ud7', 'ud8', 'ud9', 'ud10', 'ud11', 'ud12', 'um0', 'um1',
    'um2', 'um3', 'um4', 'um5', 'um6', 'um7', 'um8', 'um9', 'um10',
    'um16', 'um15', 'um17', 'um11', 'um12', 'um13', 'um14', 'm0', 'm1',
    'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11',
    'm12', 'm13', 'm14', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
    'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'cd1', 'cd2', 'cd3',
    'cd4', 'cd5', 'cd6', 'cd7', 'dr1', 'dr2', 'dr3', 'dr4', 'dr5',
    'dr6', 'dr7', 'ou1', 'ou2', 'ou3', 'ou4']

label = ['Label']

features_pipeline = Pipeline([
    ('features', FeatureUnion([
        ('continuous', Pipeline([
            ('extract', ColumnSelector(continous)),
            ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
            ('normalize', Normalizer())
        ])),
    ])),
])

features_pipeline.fit(dataset_beta, dataset_beta.Label.values.ravel())

train_dataset_x = features_pipeline.transform(dataset_beta)
train_dataset_y = dataset_beta.Label.values.ravel()

valid_dataset_x = features_pipeline.transform(dataset_alpha)
# valid_dataset_y = dataset_alpha.Label.values.ravel()

train_dmatrix = xgb.DMatrix(train_dataset_x, label=train_dataset_y)

# xgboost模型训练
params = {
  'booster': 'gbtree',
  'objective': 'binary:logistic',
  'eval_metric': 'auc',
  'gamma': 0.1,
  'min_child_weight': 1.1,
  'max_depth': 12,
  'lambda': 10,
  'subsample': 0.7,
  'colsample_bytree': 0.7,
  'colsample_bylevel': 0.7,
  'eta': 0.01,
  'seed': 0,
  'nthread': 4,
}

# 使用xgb.cv优化num_boost_round参数
cvresult = xgb.cv(params, train_dmatrix, num_boost_round=10000, nfold=2, metrics='auc', seed=0, callbacks=[
    xgb.callback.print_evaluation(show_stdv=False),
    xgb.callback.early_stop(10)
])
num_round_best = cvresult.shape[0] - 1
print('Best round num: ', num_round_best)