
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.ensemble import StackingClassifier

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

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 使用 SMOTE + Tomek Links 组合方法进行数据不平衡处理
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
# print(y.astype())
# 一级模型列表
base_models = [
    ('svm', SVC(probability=True, kernel='rbf', C=33.048, gamma=0.1539, random_state=0)),
    ('dt', DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=18, min_samples_leaf=7,random_state=0, )),
    ('rf',RandomForestClassifier(n_estimators=189, max_depth=18, min_samples_split=5, min_samples_leaf=2, random_state=0)),
    ('xgb', XGBClassifier(use_label_encoder=False,eval_metric='mlogloss', max_depth=7,eta=0.2469,gamma=0.2654,n_estimators=100,random_state=0))
]

# 使用逻辑回归作为元模型
meta_model = LogisticRegression(multi_class='multinomial', solver='lbfgs',penalty='l2', C=28.8678, random_state=0)
# 创建StackingClassifier

stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model)

# 训练Stacking模型
stacking_clf.fit(X_resampled, y_resampled)

# 预测测试集
y_pred = stacking_clf.predict(X_test)

# 评估性能
print("Stacking Model Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))

#模型参数优化
def percept(args):
    args["max_depth"] = int(args["max_depth"])
    # 一级模型列表
    base_models = [
        ('svm', SVC(probability=True, kernel='rbf', C=args["C"], gamma=args["gamma"], random_state=0)),
        ('dt', DecisionTreeClassifier(criterion='entropy', max_depth=int(args["max_depth0"]), min_samples_split=int(args["min_samples_split0"]), min_samples_leaf=int(args["min_samples_leaf0"]),
                                      random_state=0, )),
        ('rf', RandomForestClassifier(n_estimators=int(args["n_estimators"]), max_depth=int(args["max_depth"]), min_samples_split=int(args["min_samples_split"]), min_samples_leaf=int(args["min_samples_leaf"]),
                                      random_state=0)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', max_depth=int(args["max_depth1"]), eta=float(args["eta"]), gamma=float(args["gamma1"]),
                              n_estimators=int(args["n_estimators1"]), random_state=0))
    ]

    # 使用逻辑回归作为元模型
    meta_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=28.8678, random_state=0)
    # 创建StackingClassifier

    stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model)

    # 训练Stacking模型
    stacking_clf.fit(X_resampled, y_resampled)
    # 预测测试集
    y_pred = stacking_clf.predict(X_test)

    return -accuracy_score(y_test, y_pred)

from hyperopt import fmin,tpe,hp,partial,Trials

space = {
    #SVM参数
         "C":hp.uniform("C",0.01,50),
         "gamma":hp.uniform("gamma",0.01,20),
    #决策树参数
         "max_depth0":hp.uniform("max_depth0",2,8),
         "min_samples_split0":hp.uniform("min_samples_split0",2,50),
         "min_samples_leaf0":hp.uniform("min_samples_leaf0",2,50),
    #随机森林参数
         "n_estimators":hp.uniform("n_estimators",2,300),
         "max_depth":hp.quniform("max_depth",2,30,1),
         "min_samples_split":hp.quniform("min_samples_split",2,121,1),
         "min_samples_leaf":hp.quniform("min_samples_leaf",2,121,1),
    #XGBoost参数
         "eta":hp.uniform("eta",0.01,0.5),
         "max_depth1":hp.uniform("max_depth1",2,20),
         "gamma1":hp.uniform("gamma1",0.001,1),
         "n_estimators1":hp.uniform("n_estimators1",2,301),
    #softmax参数
         "C2":hp.uniform("C2",0.01,50)
         }

algo = partial(tpe.suggest,n_startup_jobs=10)
best = fmin(percept,space,algo = algo,max_evals=200)
print(best)
print(percept(best))


