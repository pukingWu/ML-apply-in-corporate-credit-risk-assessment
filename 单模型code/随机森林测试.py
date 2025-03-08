import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek

file = r'C:\Users\吴振波\Desktop\mltest1.csv'
cred = pd.read_csv(file)
pd.set_option('future.no_silent_downcasting', True)
cred.replace({'企业信用等级':{'C':1,'CC':2,'CCC':3,'B':4,'BB':5,'BBB':6,'A':7,'AA':8,'AAA':9}})
X = cred.drop(columns = ['企业信用等级'])
y = cred['企业信用等级']


# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# 使用 SMOTE + Tomek Links 组合方法进行数据不平衡处理
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# 初始化随机森林分类器，这里我们使用100棵树
rf = RandomForestClassifier(n_estimators=193, max_depth=23,min_samples_split=5,min_samples_leaf=2,random_state=0)

# 使用训练数据拟合模型
rf.fit(X_resampled, y_resampled)

# 使用模型对测试集进行预测
y_pred = rf.predict(X_test)
# 预测测试集
# y_pred = meta_model.predict(S_test_flat)
#获取预测概率
y_pred_proba = rf.predict_proba(X_test)  # 获取概率预测
# 计算每个类别的ROC AUC，然后计算平均值（宏平均或微平均）
from sklearn.metrics import roc_auc_score
macro_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')  # 宏平均
print("宏平均ROC AUC分数为:", macro_roc_auc)

# 计算预测准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型在测试集上的准确率：{accuracy}")

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
macro_f1 = f1_score(y_test, y_pred, zero_division=1,average="macro")
print("f1指数为：",macro_f1)

#{'max_depth': 18.56589199552943, 'min_samples_leaf': 2.078402640637945, 'min_samples_split': 4.4877619653998035, 'n_estimators': 189.1727868034622}
# 模型在测试集上的准确率：0.4855072463768116
# Kappa系数为： 0.3437813944143058
# 宏查准率为: 0.44483529129549665
# 宏查全率为: 0.5358090953853386
# f1指数为： 0.46832098473830136
#{'max_depth': 15.747257787645866, 'min_samples_leaf': 2.7129972743163187, 'min_samples_split': 9.376750756842279, 'n_estimators': 91.70340509708423}

#{'max_depth': 14.033564908622855, 'min_samples_leaf': 2.266982154714975, 'min_samples_split': 3.0483698189036748, 'n_estimators': 488.6991892656083}

#{'max_depth': 14.721025620639889, 'min_samples_leaf': 4.319546002138075, 'min_samples_split': 2.2573610945811673, 'n_estimators': 144.83875519573232}

#{'max_depth': 14.0, 'min_samples_leaf': 3.0, 'min_samples_split': 28.0, 'n_estimators': 99.87878184994491}


# {'max_depth': 12.0, 'min_samples_leaf': 2.0, 'min_samples_split': 2.0, 'n_estimators': 299.5938108896662}
# 模型在测试集上的准确率：0.44565217391304346
# Kappa系数为： 0.3174131935433643
# 宏查准率为: 0.4075083670228583
# 宏查全率为: 0.5096531085314866
# f1指数为： 0.4301929585792074

# {'max_depth': 23.0, 'min_samples_leaf': 2.0, 'min_samples_split': 5.0, 'n_estimators': 193.682847113695}
# 模型在测试集上的准确率：0.4746376811594203
# Kappa系数为： 0.33860514707419564
# 宏查准率为: 0.4373744804072673
# 宏查全率为: 0.5318248584774091
# f1指数为： 0.46235758152264844

#{'max_depth': 22.0, 'min_samples_leaf': 2.0, 'min_samples_split': 7.0, 'n_estimators': 135.46508799225595}
# 模型在测试集上的准确率：0.44565217391304346
# Kappa系数为： 0.2482910851609228
# 宏查准率为: 0.4214385451150157
# 宏查全率为: 0.5167756919180521
# f1指数为： 0.4423456132951868

# {'max_depth': 30.0, 'min_samples_leaf': 2.0, 'min_samples_split': 7.0, 'n_estimators': 47.540154307478424}
# 模型在测试集上的准确率：0.4492753623188406
# Kappa系数为： 0.2662214785042927
# 宏查准率为: 0.4176231981480533
# 宏查全率为: 0.5159840902593704
# f1指数为： 0.4403038634351037