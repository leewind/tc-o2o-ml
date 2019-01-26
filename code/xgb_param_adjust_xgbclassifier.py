# -*- coding: UTF-8 -*-

xgb_min_num_round = 10
xgb_max_num_round = 500
xgb_num_round_step = 10

xgb_random_seed = 2018
xgb_nthread = 4
xgb_dmatrix_silent = True

space = {
    'booster': 'gblinear',
    'objective': 'rank:pairwise',
    'nthread': xgb_nthread,
    'silent' : True,
    'seed': xgb_random_seed,
    "max_evals": 200,
    'eval_metric': 'auc',
    'max_depth': hp.quniform('max_depth', 6, 18, 1),
    'eta' : hp.quniform('eta', 0.01, 1, 0.01),
#     'lambda' : hp.quniform('lambda', 0, 5, 0.05),
#     'alpha' : hp.quniform('alpha', 0, 0.5, 0.005),
#     'lambda_bias' : hp.quniform('lambda_bias', 0, 3, 0.1),
#     'num_round' : hp.quniform('num_round', xgb_min_num_round, xgb_max_num_round, xgb_num_round_step),
    'n_estimators': hp.quniform('n_estimators', 100, 500, 50),
}

watchlist = [(train_dataset_x, train_dataset_y), (valid_dataset_x, valid_dataset_y)]

def objective(params):
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""

    logger.info(params)
    bst = xgb.sklearn.XGBClassifier(
        nthread=params['nthread'],
        learn_rate=params['eta'],
        max_depth=int(params['max_depth']),
        min_child_weight=1.1,
        subsample=0.7,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        objective=params['objective'],
        n_estimators=int(params['n_estimators']),
        gamma=0.1,
        reg_alpha=0,
        reg_lambda=1,
        max_delta_step=0,
        scale_pos_weight=1,
        silent=params['silent']
    )
    bst.fit(train_dataset_x, train_dataset_y, eval_set=watchlist, eval_metric=params['eval_metric'], early_stopping_rounds=10)

    predict_test_prob_y = bst.predict_proba(valid_dataset_x)
    model_test_df['Probability'] = predict_test_prob_y[:, 1]
    score = evaluate(model_test_df)
    logging.info('Socre is %f' % score)

    # Loss must be minimized
    loss = 1 - score

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

MAX_EVALS = 200

# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = Trials())
best