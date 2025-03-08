import rand as rand
from sklearn import datasets
import numpy as np
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=123)

# 使用 SMOTE + Tomek Links 组合方法进行数据不平衡处理
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

#数据标准化
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=0.01, random_state=42)
model.fit(X_resampled,y_resampled)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

def percept(args):
    global X_train,y_train,y_test
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=float(args['C']), random_state=0)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    return -accuracy_score(y_test, y_pred)

from hyperopt import fmin,tpe,hp,partial

space = {"penalth":hp.choice('penalth',['l1','l2']),
         "C":hp.uniform("C",0.01,100)
         }
algo = partial(tpe.suggest,n_startup_jobs=10)
best = fmin(percept,space,algo = algo,max_evals=200)
print(best)
print(percept(best))
#0.822222222222
#{'n_iter': 14, 'eta': 0.12877033763511717}
#-0.911111111111