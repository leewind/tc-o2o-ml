params = {
    # gbtree and dart use tree based models while gblinear uses linear functions.
    'booster': 'gbtree',
    #  Use LambdaMART to perform pairwise ranking where the pairwise loss is minimized
    'objective': 'rank:pairwise',
    # auc: Area under the curve
    'eval_metric': 'auc',
    # Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be.
    'gamma': 0.1,
    'min_child_weight': 1.1,

    'max_depth':4,
#     'max_depth': 12,
    # Maximum number of nodes to be added
    'max_leaves': 128,
    # L2 regularization term on weights. Increasing this value will make model more conservative.
    'lambda': 3,

    'alpha': 2,
    # Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting.
    'subsample': 0.7,
    # This is a family of parameters for subsampling of columns.
    'colsample_bytree': 0.7,
    'colsample_bylevel': 0.7,
    # learning_rate
    'eta': 0.01,
    # Exact greedy algorithm
    'tree_method': 'exact',
    # Random number seed.
    'seed': 0,
    'nthread': 4,
    # Verbosity of printing messages. Valid values are 0 (silent),
    'verbosity': 0,
    'metric_freq': 100,
}

watchlist = [(xgbtrain, 'train'), (xgbvalid, 'validate')]

logging.info('train begin')
model = xgb.train(params, xgbtrain, num_boost_round=200, evals=watchlist)
logging.info('train end')
model.save_model('../model/xgb.model')