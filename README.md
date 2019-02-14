# [天池新人实战赛o2o优惠券使用预测](https://tianchi.aliyun.com/competition/entrance/231593/introduction)

## HOWTO 如何运行

1. 创建`source`目录，从[网站](https://tianchi.aliyun.com/competition/entrance/231593/information)下载比赛数据到目录中
2. 创建`features`目录，用来存放生成的特征数据
3. 运行`o2o_coupon_prediction_features_v2.ipynb`，在`数据集划分`中修改数据集信息 ，执行`run all cells`后会生成特征文件在`features`目录下，生成全部数据集需要执行三次
4. 运行`o2o_coupon_prediction_model_*.ipynb`查看预测结果


思路

+ 考虑使用[mlxtend](https://rasbt.github.io/mlxtend/)的`EnsembleVoteClassifier`的方案进行试验
