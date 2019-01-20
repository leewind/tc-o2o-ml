# 天池新人实战赛o2o优惠券使用预测

+ `/code/generate_features.py`执行之后会在`features`文件夹中生成特征文件`lcm_*.csv`
+ 使用`jupyter`目录中的模型进行预测，当前模型是`o2o_coupon_prediction_gbdt_lr.ipynb`


思路

+ 考虑使用[mlxtend](https://rasbt.github.io/mlxtend/)的`EnsembleVoteClassifier`的方案进行试验