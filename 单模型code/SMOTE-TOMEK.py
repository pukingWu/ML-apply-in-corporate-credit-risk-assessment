from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# # 创建一个不平衡的二分类数据集
# X, y = make_classification(n_classes=2, class_sep=2,
#                            weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
#                            n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
import pandas as pd
file = r'C:\Users\吴振波\Desktop\实验数据集_OpenMLselect3.csv'
cred = pd.read_csv(file)
pd.set_option('future.no_silent_downcasting', True)
cred.replace({'企业信用等级':{'C':1,'CC':2,'CCC':3,'B':4,'BB':5,'BBB':6,'A':7,'AA':8,'AAA':9}})

# 2.提取特征变量与目标变量
X = cred.drop(columns = ['企业信用等级'])
y = cred['企业信用等级']

# 划分训练集与测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# 使用 SMOTE + Tomek Links 组合方法
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
# print(X_resampled)
# print(y_resampled)

# 使用随机森林作为分类器
clf = RandomForestClassifier(random_state=42)
clf.fit(X_resampled, y_resampled)

# 在测试集上进行预测
y_pred = clf.predict(X_test)
