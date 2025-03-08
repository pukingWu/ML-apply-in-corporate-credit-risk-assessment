import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split

file = r'C:\Users\吴振波\Desktop\实验数据集_OpenMLselect2.csv'
cred = pd.read_csv(file)
cred_df = pd.DataFrame(cred)
pd.set_option('future.no_silent_downcasting', True)
# a3 = pd.get_dummies(cred_df, columns=['企业信用等级']).astype(int)
# print(a3)
cred.replace({'企业信用等级':{'C':1,
                'CC':2,
                'CCC':3,
                'B':4,
                'BB':5,
                'BBB':6,
                'A':7,
                'AA':8,
                'AAA':9}})

X = cred.drop(columns = ['企业信用等级'])
y = cred['企业信用等级']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建SVM模型
svm = SVC(kernel="linear", C=1)

# 创建RFE对象，设置要选择的特征数量
rfe = RFE(estimator=svm, n_features_to_select=12, step=1)

# 使用RFE进行特征选择
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

# 打印出选择的特征索引
print("Num Features: %s" % (X_train_rfe.shape[1]))
print("Selected Features: %s" % (rfe.support_))
print("Feature Ranking: %s" % (rfe.ranking_))