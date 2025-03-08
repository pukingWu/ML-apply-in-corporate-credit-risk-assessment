from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from imblearn.combine import SMOTETomek
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

file = r'C:\Users\吴振波\Desktop\mltest1.csv'
cred = pd.read_csv(file)
pd.set_option('future.no_silent_downcasting', True)
cred.replace({'企业信用等级':{'C':1,'CC':2,'CCC':3,'B':4,'BB':5,'BBB':6,'A':7,'AA':8,'AAA':9}})


# 2.提取特征变量与目标变量
X = cred.drop(columns = ['企业信用等级'])
y = cred['企业信用等级']

# label_encoder = LabelEncoder()
# # 使用fit_transform方法将文本标签转换为整数
# integer_y = label_encoder.fit_transform(y)
#
# #将标签转化为one-hot编码
# y_one_hot = to_categorical(integer_y,num_classes=9)

# 3.划分训练集与测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# 使用 SMOTE + Tomek Links 组合方法进行数据不平衡处理
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# 初始化逻辑回归模型，并设置L2正则化
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=37.4620, random_state=0)

# 'C' 参数控制正则化的强度。较小的值指定更强的正则化

# 训练模型
clf.fit(X_resampled, y_resampled)

# 使用模型进行预测
y_pred = clf.predict(X_test)

y_pred_proba = clf.predict_proba(X_test)  # 获取概率预测
# 计算每个类别的ROC AUC，然后计算平均值（宏平均或微平均）
from sklearn.metrics import roc_auc_score
macro_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')  # 宏平均
print("宏平均ROC AUC分数为:", macro_roc_auc)
# 计算预测的准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 查看模型的系数和截距
# print("Coefficients: \n", clf.coef_)
# print("Intercept: \n", clf.intercept_)

from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_test,y_pred) #(label除非是你想计算其中的分类子集的kappa系数，否则不需要设置)
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

#{'C': 44.29575864, 'penalth': 0}
# Accuracy: 0.3007246376811594
# Kappa系数为： 0.1583770460721735
# 宏查准率为: 0.3018684835193231
# 宏查全率为: 0.39257575075906437
# f1指数为： 0.2843921472678303

# {'C': 39.955856379033285, 'penalth': 0}
# Accuracy: 0.2898550724637681
# Kappa系数为： 0.1465354032563423
# 宏查准率为: 0.2928818445836292
# 宏查全率为: 0.38308618167007596
# f1指数为： 0.2736386128491392

# {'C': 30.86120874588547, 'penalth': 0}
# Accuracy: 0.2898550724637681
# Kappa系数为： 0.14322368108459127
# 宏查准率为: 0.28879082649196725
# 宏查全率为: 0.3796077320108413
# f1指数为： 0.27243363000047954

# {'C': 37.462028947608935, 'penalth': 0}
# Accuracy: 0.30434782608695654
# Kappa系数为： 0.16045627376425853
# 宏查准率为: 0.3055986228117671
# 宏查全率为: 0.39180669115046385
# f1指数为： 0.2856533248096937

# {'C': 64.9387954911376, 'penalth': 0}
# Accuracy: 0.29347826086956524
# Kappa系数为： 0.1517463119404866
# 宏查准率为: 0.30095302315873124
# 宏查全率为: 0.3874013405694566
# f1指数为： 0.27757715054231136