import rand as rand
from sklearn import datasets
import numpy as np
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
import pandas as pd
file = r'C:\Users\吴振波\Desktop\实验数据集_OpenMLselect3.csv'
cred = pd.read_csv(file)
pd.set_option('future.no_silent_downcasting', True)
# cred.replace({'企业信用等级':{'C':1,'CC':2,'CCC':3,'B':4,'BB':5,'BBB':6,'A':7,'AA':8,'AAA':9}})

# 2.提取特征变量与目标变量
X = cred.drop(columns = ['企业信用等级'])
y = cred['企业信用等级']

label_encoder = LabelEncoder()
# 使用fit_transform方法将文本标签转换为整数
integer_y = label_encoder.fit_transform(y)

# 因为我们有九个类别，所以我们需要将标签转换为one-hot编码
y_one_hot = to_categorical(integer_y, num_classes=9)
# 3.划分训练集与测试集
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y_one_hot,test_size=0.2,random_state=123)
smote_tomek = SMOTETomek(random_state=0)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)
model = Sequential()
model.add(Dense(10, input_dim=13, activation='leaky_relu'))  # 输入层到隐藏层，假设有10个神经元
model.add(Dense(9, activation='softmax'))  # 隐藏层到输出层，9个神经元对应9个类别，使用softmax激活函数

# 编译模型，设置优化器、损失函数和评估指标
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=40, batch_size=10, verbose=1)

# 评估模型
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# 使用模型进行预测
y_pred = model.predict(X_test)
# 因为预测结果是概率分布，所以我们需要取概率最大的类别作为预测结果
y_pred_classes = np.argmax(y_pred, axis=1)
# 输出预测结果（如果需要）
print("Predicted classes:", y_pred_classes)

def percept(args):
    global X_train,y_train,y_test
    model = Sequential()
    model.add(Dense(int(args["n_p"]), input_dim=13, activation='leaky_relu'))  # 输入层到隐藏层，假设有10个神经元
    model.add(Dense(9, activation='softmax'))  # 隐藏层到输出层，9个神经元对应9个类别，使用softmax激活函数
    # 编译模型，设置优化器、损失函数和评估指标
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 训练模型
    model.fit(X_train, y_train, epochs=int(args["epochs"]), batch_size=int(args["batch_size"]), verbose=1)
    # model = DecisionTreeClassifier(criterion='entropy',max_depth=int(args["max_depth"]),min_samples_split=int(args["min_samples_split"]),min_samples_leaf=int(args["min_samples_leaf"]),random_state=0)
    # model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy

from hyperopt import fmin,tpe,hp,partial

space = {
         "epochs":hp.uniform("epochs",1,200),
         "batch_size":hp.uniform("batch_size",5,100),
         "n_p":hp.uniform("n_p",2,50),
        }

algo = partial(tpe.suggest,n_startup_jobs=10)
best = fmin(percept,space,algo = algo,max_evals=200)
print(best)
print(percept(best))
#0.822222222222
#{'n_iter': 14, 'eta': 0.12877033763511717}
#-0.911111111111