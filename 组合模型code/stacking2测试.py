import numpy as np
from sklearn.datasets import make_classification  # 这行在您的实际代码中可能不需要
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.ensemble import StackingClassifier
from imblearn.combine import SMOTETomek

file = r'C:\Users\吴振波\Desktop\mltest1.csv'
cred = pd.read_csv(file)
pd.set_option('future.no_silent_downcasting', True)
# 注意：您应该使用 inplace=True 或重新赋值来确保替换是永久性的
cred.replace({'企业信用等级': {'C': 6, 'CC': 7, 'CCC': 0, 'B': 1, 'BB': 2, 'BBB': 3, 'A': 4, 'AA': 5, 'AAA': 9}},
             inplace=True)
X = cred.drop(columns=['企业信用等级'])
y = cred['企业信用等级'].astype(int)

# # 编码标签为整数（如果它们还不是）
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y.astype(str))
# 假设我们要进行5折分层交叉验证

# skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# 使用 SMOTE + Tomek Links 组合方法进行数据不平衡处理
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
# print(y.astype())
# 一级模型列表
base_models = [
    ('svm', SVC(probability=True, kernel='rbf', C=15.9470, gamma=10.9539, random_state=0)),
    ('dt', DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=8, min_samples_leaf=6,random_state=0, )),
    ('rf',RandomForestClassifier(n_estimators=155, max_depth=11, min_samples_split=71, min_samples_leaf=61, random_state=0)),
    ('xgb', XGBClassifier(use_label_encoder=False,eval_metric='mlogloss', max_depth=14,eta=0.2815,gamma=0.0476,n_estimators=95,random_state=0))
]
# {'C': 19.527363707352784, 'C2': 28.86789707938741, 'eta': 0.47955927173855306, 'gamma': 14.635896344460855, 'gamma1': 0.044068829322820025, 'max_depth': 19.0, 'max_depth0': 7.670697326227156, 'max_depth1': 15.031545572414258, 'min_samples_leaf': 72.0, 'min_samples_leaf0': 30.425722506977916, 'min_samples_split': 44.0, 'min_samples_split0': 33.6813676515981, 'n_estimators': 208.45549749084304, 'n_estimators1': 213.18771500423185}
#{'C': 15.947032431909541, 'eta': 0.28150958439970614, 'gamma': 10.953943491158942, 'gamma1': 0.04761058335290208, 'max_depth': 11.0, 'max_depth0': 6.014643928222784, 'max_depth1': 14.89046036390142, 'min_samples_leaf': 61.0, 'min_samples_leaf0': 5.644981388481787, 'min_samples_split': 71.0, 'min_samples_split0': 8.401819394451547, 'n_estimators': 155.3886704191565, 'n_estimators1': 95.59999083315412}
# Stacking Model Accuracy: 0.53
# Kappa系数为： 0.6493185495478504
# 宏查准率为: 0.5204808419094133
# 宏查全率为: 0.5098559948093172
# f1指数为： 0.5131918128641274
# 使用逻辑回归作为元模型
meta_model = LogisticRegression(multi_class='multinomial', solver='lbfgs',penalty='l2', C=1, random_state=0)
# 创建StackingClassifier

stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# 训练Stacking模型
stacking_clf.fit(X_resampled, y_resampled)
y_pred = stacking_clf.predict(X_test)

# Lift for Class 0: 11.50
# Lift for Class 1: 2.30
# Lift for Class 2: 2.09
# Lift for Class 3: 1.72
# Lift for Class 4: 3.26
# Lift for Class 5: 17.25
import matplotlib.pyplot as plt
# 计算每个类别的真实和预测的正样本数量
n_samples_test = len(y_test)
class_counts = np.bincount(y_test)
class_prob_overall = class_counts / n_samples_test

# 初始化lift数组
lift_scores = []

# 计算每个类别的lift
for class_idx in range(len(class_prob_overall)):
    class_mask = y_test == class_idx
    class_samples = np.sum(class_mask)

    # 计算该类别的预测为正样本的数量
    predicted_positives = np.sum(y_pred[class_mask] == class_idx)

    # 计算该类别的响应率（真实正样本率）
    response_rate_targeted = predicted_positives / class_samples if class_samples > 0 else 0

    # 计算lift
    lift_score = response_rate_targeted / class_prob_overall[class_idx]
    lift_scores.append(lift_score)

# 绘制lift图
classes = np.arange(len(class_prob_overall))
plt.bar(classes, lift_scores)
# 添加数值标注

for idx, lift in enumerate(lift_scores):
    plt.text(idx, lift + 0.1,  # 稍微向上偏移一些，避免与柱子重叠
             f'{lift:.2f}',  # 保留两位小数
             ha='center')  # 水平居中对齐
plt.title('Lift Scores for Each Class')
plt.xlabel('Class Index')
plt.ylabel('Lift Score')
plt.xticks(classes, [f'Class {i}' for i in classes])
plt.show()

# 打印每个类别的lift分数
for class_idx, lift_score in enumerate(lift_scores):
    print(f"Lift for Class {class_idx}: {lift_score:.2f}")
# 拟合元模型
# 获取元模型的系数和截距项
# meta_estimator = stacking_clf.final_estimator_   # 获取元模型，即LogisticRegression对象
# 输出系数
#2.926   0.4367   0.2022   1.5711
# [-0.44066006  0.1101595   0.58053226  0.73272558  0.00480081 -0.98755809]
# coef = meta_estimator.coef_
# print("Coefficients:")
# # co = pd.DataFrame(coef)
# for i, coef_i in enumerate(coef):
#     print(f"Class {i}: {coef_i}")
# # 输出截距项
# intercept = meta_estimator.intercept_
# print("Intercept:")
# print(intercept)

# 预测测试集
# y_pred = stacking_clf.predict(X_test)
# #获取预测概率
# y_pred_proba = stacking_clf.predict_proba(X_test)  # 获取概率预测
# # 计算每个类别的ROC AUC，然后计算平均值（宏平均或微平均）
# from sklearn.metrics import roc_auc_score
# macro_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')  # 宏平均
# print("宏平均ROC AUC分数为:", macro_roc_auc)
#
# # 评估性能
# print("Stacking Model Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
#
# # from sklearn.metrics import cohen_kappa_score
# kappa = cohen_kappa_score(y_test,y_pred,weights='quadratic') #(label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
# print("Kappa系数为：",kappa)
# #
# # from sklearn.metrics import precision_score,recall_score,f1_score
# #计算宏查准率
# macro_precision = precision_score(y_test, y_pred, zero_division=1,average="macro")
# print("宏查准率为:",macro_precision)
# #计算宏查全率
# macro_recall = recall_score(y_test, y_pred, zero_division=1,average="macro")
# print("宏查全率为:",macro_recall)
# #计算f1指数
# macro_f1 = f1_score(y_test, y_pred, zero_division=1,average="macro")
# print("f1指数为：",macro_f1)

# 100%|██████████| 10/10 [01:19<00:00,  7.94s/trial, best loss: -0.5217391304347826]
# {'C': 11.93337968403566, 'C2': 13.369733797629104, 'eta': 0.03695988162736242, 'gamma': 1.3387537501477873, 'gamma1': 0.7881601470618914, 'max_depth': 28.0, 'max_depth0': 2.6529450283400746, 'max_depth1': 15.606448351676768, 'min_samples_leaf': 5.0, 'min_samples_leaf0': 45.7625968847135, 'min_samples_split': 115.0, 'min_samples_split0': 17.482302760887766, 'n_estimators': 168.54933773618671, 'n_estimators1': 139.77716211539368}
# l Accuracy: 0.50
# Kappa系数为： 0.6158682841603411
# 宏查准率为: 0.48032984898342534
# 宏查全率为: 0.4547730846012219
# f1指数为： 0.4613783085362262


# 100%|██████████| 100/100 [15:19<00:00,  9.19s/trial, best loss: -0.5362318840579711]
# {'C': 23.145698351111996, 'C2': 45.39391905945338, 'eta': 0.04812407492852991, 'gamma': 16.61123370447934, 'gamma1': 0.06161212364097393, 'max_depth': 7.0, 'max_depth0': 3.5975926575361177, 'max_depth1': 14.726362015206886, 'min_samples_leaf': 52.0, 'min_samples_leaf0': 18.84900744189204, 'min_samples_split': 26.0, 'min_samples_split0': 25.96980214211299, 'n_estimators': 175.4088316871341, 'n_estimators1': 213.88947137332428}
# l Accuracy: 0.53
# Kappa系数为： 0.6418638357881542
# 宏查准率为: 0.518098035652999
# 宏查全率为: 0.5239542566857607
# f1指数为： 0.5166168680744779

