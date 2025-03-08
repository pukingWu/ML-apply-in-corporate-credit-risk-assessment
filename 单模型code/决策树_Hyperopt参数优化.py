
from sklearn import datasets
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()

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

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',max_depth=3,min_samples_split=3,min_samples_leaf=3,random_state=0)
model.fit(X_resampled,y_resampled)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

def percept(args):
    global X_resampled,y_resampled,y_test
    model = DecisionTreeClassifier(criterion='entropy',max_depth=int(args["max_depth"]),min_samples_split=int(args["min_samples_split"]),min_samples_leaf=int(args["min_samples_leaf"]),random_state=0)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    return -accuracy_score(y_test, y_pred)

from hyperopt import fmin,tpe,hp,partial

space = {
    # "criterion":hp.choice("criterion",['gini','entropy']),
         "max_depth":hp.uniform("max_depth",2,30,),
         "min_samples_split":hp.uniform("min_samples_split",2,100),
         "min_samples_leaf":hp.uniform("min_samples_leaf",2,100)}
algo = partial(tpe.suggest,n_startup_jobs=10)
best = fmin(percept,space,algo = algo,max_evals=200)
print(best)
print(percept(best))
#0.822222222222
#{'n_iter': 14, 'eta': 0.12877033763511717}
#-0.911111111111