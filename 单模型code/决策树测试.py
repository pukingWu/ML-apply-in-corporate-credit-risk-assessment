# 1.数据读取与预处理
from imblearn.combine import SMOTETomek
import pandas as pd
file = r'C:\Users\吴振波\Desktop\mltest1.csv'
cred = pd.read_csv(file)
pd.set_option('future.no_silent_downcasting', True)
cred.replace({'企业信用等级':{'C':1,'CC':2,'CCC':3,'B':4,'BB':5,'BBB':6,'A':7,'AA':8,'AAA':9}})

# 2.提取特征变量与目标变量
X = cred.drop(columns = ['企业信用等级'])
y = cred['企业信用等级']

# 3.划分训练集与测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# 使用 SMOTE + Tomek Links 组合方法进行数据不平衡处理
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# 4.模型训练与拟合
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',max_depth=10,min_samples_split=13,min_samples_leaf=8,random_state=0)
model.fit(X_train,y_train)

#5.模型预测
y_pred = model.predict(X_test)
#获取预测概率
y_pred_proba = model.predict_proba(X_test)  # 获取概率预测
# 计算每个类别的ROC AUC，然后计算平均值（宏平均或微平均）
from sklearn.metrics import roc_auc_score
macro_roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')  # 宏平均
print("宏平均ROC AUC分数为:", macro_roc_auc)

#6.查看模型预测准确度
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred,y_test)
print("精确率为:",score)

# #查看每个企业预测不同信用等级的概率
# y_pred_proba = model.predict_probe(X_test)  #九维数组
# b = pd.DataFrame(y_pred_proba,columns = ['C','CC','CCC','B','BB','BBB','A','AA','AAA'])
# print(b)
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

#更换数据集
#{'criterion': 0, 'max_depth': 7.347674780721415, 'min_samples_leaf': 11.271718403566089, 'min_samples_split': 19.942133392783546}
#精确率为: 0.40794223826714804
# Kappa系数为： 0.1840796019900498
# 宏查准率为: 0.5284539435838592
# 宏查全率为: 0.23289533522719744
# f1指数为： 0.2314605462564646

#设置random_state=0
#{'criterion': 1, 'max_depth': 7.967204043877793, 'min_samples_leaf': 7.051729487742136, 'min_samples_split': 11.611777756370406}
# 精确率为: 0.43478260869565216
# Kappa系数为： 0.24307789673540425
# 宏查准率为: 0.5101756609499972
# 宏查全率为: 0.3677314920611126
# f1指数为： 0.3983893410021751

#进行了数据不平衡处理
# {'criterion': 0, 'max_depth': 7.675878906816297, 'min_samples_leaf': 5.598996024213479, 'min_samples_split': 19.866063602438743}
#7 7 18
# 精确率为: 0.43478260869565216
# Kappa系数为： 0.24307789673540425
# 宏查准率为: 0.5101756609499972
# 宏查全率为: 0.3677314920611126
# f1指数为： 0.3983893410021751


# {'criterion': 1, 'max_depth': 7.121187795113717, 'min_samples_leaf': 6.158768153136412, 'min_samples_split': 4.543007124965809}
# 精确率为: 0.4166666666666667
# Kappa系数为： 0.2156461264187245
# 宏查准率为: 0.33343030865519624
# 宏查全率为: 0.3158607437617929
# f1指数为： 0.3209622902616954

# {'criterion': 1, 'max_depth': 7.317933366495304, 'min_samples_leaf': 6.035605165076723, 'min_samples_split': 17.397471536958953}
# 确率为: 0.427536231884058
# Kappa系数为： 0.23160417254017485
# 宏查准率为: 0.3434962983876028
# 宏查全率为: 0.326064825394446
# f1指数为： 0.331516045735061

# {'max_depth': 7.131373908582432, 'min_samples_leaf': 6.409696826359195, 'min_samples_split': 11.955652522881062}
# 精确率为: 0.4166666666666667
# Kappa系数为： 0.2156461264187245
# 宏查准率为: 0.33343030865519624
# 宏查全率为: 0.3158607437617929
# f1指数为： 0.3209622902616954

# {'max_depth': 7.9970218413050915, 'min_samples_leaf': 5.30769523807349, 'min_samples_split': 11.23511917376451}
# 确率为: 0.3804347826086957
# Kappa系数为： 0.16350295102887213
# 宏查准率为: 0.251809625968918
# 宏查全率为: 0.2605529903015647
# f1指数为： 0.2550445107163397


#更改参数
# {'max_depth': 11.54293793991936, 'min_samples_leaf': 5.048707502162073, 'min_samples_split': 10.838050050829711}

# {'max_depth': 10.771836031497692, 'min_samples_leaf': 8.161160331451756, 'min_samples_split': 19.84368373538109}
# 精确率为: 0.42028985507246375
# Kappa系数为： 0.24510239666313383
# 宏查准率为: 0.3878409366524816
# 宏查全率为: 0.4203132248384463
# f1指数为： 0.38934071094556794

# {'max_depth': 10.699220058211258, 'min_samples_leaf': 9.845037739068072, 'min_samples_split': 13.624247606159965}