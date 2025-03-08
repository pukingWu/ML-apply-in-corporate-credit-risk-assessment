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
rf = RandomForestClassifier(n_estimators=100, max_depth=8,min_samples_split=18,min_samples_leaf=7,random_state=0)

# 使用训练数据拟合模型
rf.fit(X_resampled, y_resampled)

# 使用模型对测试集进行预测
y_pred = rf.predict(X_test)

# 计算预测准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型在测试集上的准确率：{accuracy}")

#模型参数优化
def percept(args):
    args["max_depth"] = int(args["max_depth"])
    model = RandomForestClassifier(n_estimators=int(args["n_estimators"]), max_depth=args["max_depth"],min_samples_split=int(args["min_samples_split"]),min_samples_leaf=int(args["min_samples_leaf"]),random_state=0)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    return -accuracy_score(y_test, y_pred)

from hyperopt import fmin,tpe,hp,partial,Trials

space = {
         "n_estimators":hp.uniform("n_estimators",2,300),
         "max_depth":hp.quniform("max_depth",2,30,1),
         "min_samples_split":hp.quniform("min_samples_split",2,121,1),
         "min_samples_leaf":hp.quniform("min_samples_leaf",2,121,1)
         }

algo = partial(tpe.suggest,n_startup_jobs=10)
best = fmin(percept,space,algo = algo,max_evals=200)
print(best)
print(percept(best))