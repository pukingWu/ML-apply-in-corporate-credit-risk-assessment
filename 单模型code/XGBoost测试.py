# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# 加载乳腺癌数据集
file = r'C:\Users\吴振波\Desktop\mltest1.csv'
cred = pd.read_csv(file)
pd.set_option('future.no_silent_downcasting', True)
cred.replace({'企业信用等级': {'C': 6, 'CC': 7, 'CCC': 0, 'B': 1, 'BB': 2, 'BBB': 3, 'A': 4, 'AA': 5, 'AAA': 9}},
             inplace=True)
X = cred.drop(columns = ['企业信用等级'])
y = cred['企业信用等级']

# 编码标签为整数（如果它们还不是）
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(str))

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
# 使用 SMOTE + Tomek Links 组合方法进行数据不平衡处理
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# 将数据转换为DMatrix格式，这是XGBoost的特定数据结构
dtrain = xgb.DMatrix(X_resampled, label=y_resampled)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置XGBoost参数
param = {
    'max_depth': 5,  # 树的深度
    'eta': 0.1692,  # 学习率
    'objective': 'multi:softmax',  # 定义学习任务及对应的学习目标
    'num_class': 6,
    'gamma':0.7037 }  # 类别数，二分类问题设置为1

num_round = 79  # 迭代次数

# 训练模型
bst = xgb.train(param, dtrain, num_round)

# 使用模型进行预测
preds = bst.predict(dtest)
# print(preds)
# 因为XGBoost的预测结果是概率，我们需要将概率转换为类别（0或1）
y_pred = preds
# # print(y_pred)
# y_test_encoded = label_encoder.transform(y_test.astype(str))
# from sklearn.metrics import roc_auc_score
# # 初始化AUC列表
# aucs = []
#
# # 对每个类别计算ROC AUC
# for i in range(6):
#     # 获取第i个类别的真实标签和预测概率
#     y_true_i = (y_test_encoded == i).astype(int)
#     y_scores_i = preds[:, i]
#
#     # 计算第i个类别的ROC AUC
#     auc = roc_auc_score(y_true_i, y_scores_i)
#     aucs.append(auc)
#
# # 计算平均AUC（宏平均）
# macro_auc = np.mean(aucs)
# print("宏平均AUC为:", macro_auc)

accuracy = accuracy_score(y_test, y_pred)
print(f"模型在测试集上的准确率：{accuracy}")

# 获取特征重要性
importance = bst.get_fscore()

# 转换特征重要性数据为DataFrame
df_importance = pd.DataFrame(list(importance.items()), columns=['feature', 'importance']).sort_values('importance',ascending=False)

# 可视化特征重要性
plt.figure(figsize=(9, 6))
plt.barh(df_importance['feature'], df_importance['importance'], align='center', color='skyblue')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title('XGBoost Feature Importances for Multi-class Classification')
plt.show()
#
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_test,y_pred,weights='quadratic') #(label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
print("Kappa系数为：",kappa)
#
from sklearn.metrics import precision_score,recall_score,f1_score
#计算宏查准率
macro_precision = precision_score(y_test, y_pred, zero_division=1,average="macro")
print("宏查准率为:",macro_precision)
#计算宏查全率
macro_recall = recall_score(y_test, y_pred, zero_division=1,average="macro")
print("宏查全率为:",macro_recall)
#计算f1指数
macro_f1 = f1_score(y_test, y_pred, zero_division=0,average="macro")
print("f1指数为：",macro_f1)

#{'eta': 0.3927644018199425, 'gamma': 0.976835835913195, 'max_depth': 12.198264338451898, 'num_around': 208.28740954013642}

#{'eta': 0.020798559380108834, 'gamma': 0.6132035822146856, 'max_depth': 11.906020654054776, 'num_around': 193.80669330574088}
# 模型在测试集上的准确率：0.47101449275362317
# Kappa系数为： 0.3580238824550772
# 宏查准率为: 0.423804491437009
# 宏查全率为: 0.49782408556686697
# f1指数为： 0.443046658981139

#{'eta': 0.3651400012316887, 'gamma': 73.63710182863535, 'max_depth': 2.0575716737667404, 'num_around': 280.36978592742014}


# 5 0.1692  0.7037  79
# 模型在测试集上的准确率：0.48188405797101447
# Kappa系数为： 0.37724582429386866
# 宏查准率为: 0.4460458412313013
# 宏查全率为: 0.5630164516734295
# f1指数为： 0.474103974626205

#{'eta': 0.49678647134460546, 'gamma': 0.5672953078180982, 'max_depth': 19.835398294108447, 'num_around': 50.235033373010374}
# 模型在测试集上的准确率：0.41304347826086957
# Kappa系数为： 0.2744973957065665
# 宏查准率为: 0.38860829389226953
# 宏查全率为: 0.45495469486251344
# f1指数为： 0.4097717248376375

#{'eta': 0.054378101287787084, 'gamma': 0.7666033482863267, 'max_depth': 13.865541392079905, 'num_around': 76.37291396295485}
# #模型在测试集上的准确率：0.4528985507246377
# Kappa系数为： 0.35709783818987795
# 宏查准率为: 0.413082592226614
# 宏查全率为: 0.48910909419248855
# f1指数为： 0.4311951693331848

# {'eta': 0.23059626243363485, 'gamma': 0.652403687731641, 'max_depth': 11.661945334297599, 'num_around': 148.44695749376402}
# 模型在测试集上的准确率：0.48188405797101447
# Kappa系数为： 0.34729137094553075
# 宏查准率为: 0.43452880842081404
# 宏查全率为: 0.5084718229226789
# f1指数为： 0.453397667284892\

# {'eta': 0.24693801568427914, 'gamma': 0.26544589450790956, 'max_depth': 7.742655648981979, 'num_around': 294.23623248254427}
# 模型在测试集上的准确率：0.4746376811594203
# Kappa系数为： 0.3887993597161984
# 宏查准率为: 0.4443108358934971
# 宏查全率为: 0.5181157052591887
# f1指数为： 0.4667631306503774