skl_min_n_estimators = 10
skl_max_n_estimators = 500
skl_n_estimators_step = 10
skl_n_jobs = 2
skl_random_seed = 2018

## random forest tree classifier
space = {
    'n_estimators': hp.quniform("n_estimators", skl_min_n_estimators, skl_max_n_estimators, skl_n_estimators_step),
    'learning_rate': hp.quniform("learning_rate", 0.01, 0.5, 0.01),
    'max_features': hp.quniform("max_features", 0.05, 1.0, 0.05),
    'max_depth': hp.quniform('max_depth', 1, 15, 1),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.1),
    'random_state': skl_random_seed
}

def objective(params):
    """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""

    logger.info(params)

    gbcf = GradientBoostingClassifier(
        n_estimators=int(params['n_estimators']),
        max_features=params['max_features'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        subsample=params['subsample'],
        random_state=params['random_state']
    )

    gbcf.fit(train_dataset_x, train_dataset_y)

    predict_test_prob_y = gbcf.predict_proba(valid_dataset_x)
    model_test_df['Probability'] = predict_test_prob_y[:, 1]

    score = evaluate(model_test_df)

#     gbdt = GradientBoostingRegressor(
#         n_estimators=int(params['n_estimators']),
#         max_features=params['max_features'],
#         learning_rate=params['learning_rate'],
#         max_depth=params['max_depth'],
#         subsample=params['subsample'],
#         random_state=params['random_state'],
#         verbose=1
#     )

#     gbdt.fit(train_dataset_x, train_dataset_y)
#     score = gbdt.score(valid_dataset_x, valid_dataset_y)
    logging.info('Socre is %f' % score)

    # Loss must be minimized
    loss = 1 - score

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}

def evaluate(result_df):
    group = result_df.groupby(['Coupon_id'])
    aucs = []
    for i in group:
        tmpdf = i[1]
        if len(tmpdf['Is_in_day_consume'].unique()) != 2:
            continue

        fpr, tpr, thresholds = roc_curve(tmpdf['Is_in_day_consume'], tmpdf['Probability'], pos_label=1)
        auc_score = auc(fpr,tpr)
        aucs.append(auc_score)

    return np.average(aucs)

MAX_EVALS = 500

# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = Trials())
best