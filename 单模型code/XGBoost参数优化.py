# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import LabelEncoder

# 加载乳腺癌数据集
file = r'C:\Users\吴振波\Desktop\mltest1.csv'
cred = pd.read_csv(file)
pd.set_option('future.no_silent_downcasting', True)
cred.replace({'企业信用等级':{'C':1,'CC':2,'CCC':3,'B':4,'BB':5,'BBB':6,'A':7,'AA':8,'AAA':9}})
X = cred.drop(columns = ['企业信用等级'])
y = cred['企业信用等级']

# 编码标签为整数（如果它们还不是）
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(str))

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)
# 使用 SMOTE + Tomek Links 组合方法进行数据不平衡处理
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# 将数据转换为DMatrix格式，这是XGBoost的特定数据结构
dtrain = xgb.DMatrix(X_resampled, label=y_resampled)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置XGBoost参数
param = {
    'max_depth': 5,  # 树的深度
    'eta': 0.1692,  # 学习率
    'objective': 'multi:softmax',  # 定义学习任务及对应的学习目标
    'num_class': 9,
    'gamma':0.7037      }  # 类别数，二分类问题设置为1

num_round = 79  # 迭代次数

# 训练模型
bst = xgb.train(param, dtrain, num_round)

# 使用模型进行预测
preds = bst.predict(dtest)
# 因为XGBoost的预测结果是概率，我们需要将概率转换为类别（0或1）
y_pred = [1 if x > 0.5 else 0 for x in preds]

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"模型在测试集上的准确率：{accuracy}")

#模型参数优化
def percept(args):
    global X_resampled,y_resampled,y_test
    # 设置XGBoost参数
    param = {
        'max_depth': int(args["max_depth"]),  # 树的深度
        'eta': float(args["eta"]),  # 学习率
        'objective': 'multi:softmax',  # 定义学习任务及对应的学习目标
        'num_class': 9,
        'gamma': float(args["gamma"])}  # 类别数，二分类问题设置为1

    num_round = int(args["num_round"])  # 迭代次数
    xgb.train(param, dtrain, num_round)
    preds = bst.predict(dtest)
    y_pred = preds
    # y_pred = [1 if x > 0.5 else 0 for x in preds]
    return -accuracy_score(y_test, y_pred)

from hyperopt import fmin,tpe,hp,partial

space = {
         "eta":hp.uniform("eta",0.01,0.5),
         "max_depth":hp.uniform("max_depth",2,20),
         "gamma":hp.uniform("gamma",0.001,1),
         "num_round":hp.uniform("num_around",2,301)
         }
algo = partial(tpe.suggest,n_startup_jobs=10)
best = fmin(percept,space,algo = algo,max_evals=200)
print(best)
print(percept(best))