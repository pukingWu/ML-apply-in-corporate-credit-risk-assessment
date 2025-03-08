import shap
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
file = r'C:\Users\吴振波\Desktop\mltest1.csv'
cred = pd.read_csv(file)
pd.set_option('future.no_silent_downcasting', True)
cred.replace({'企业信用等级': {'C': 6, 'CC': 7, 'CCC': 0, 'B': 1, 'BB': 2, 'BBB': 3, 'A': 4, 'AA': 5, 'AAA': 9}},
             inplace=True)
X = cred.drop(columns = ['企业信用等级'])
y = cred['企业信用等级']
feature_names = X.columns

# 编码标签为整数（如果它们还不是）
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.astype(int))
# data = load_iris()
# X, y = data.data, data.target
# feature_names = cred.feature_names
# target_names = cred.target_names
#
X = pd.DataFrame(X, columns=feature_names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #数据不平衡处理
# smote_tomek = SMOTETomek(random_state=0)
# X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)

# 将数据转换为DMatrix格式，这是XGBoost的特定数据结构
dtrain = xgb.DMatrix(X_train, label=y_train)
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

explainer = shap.TreeExplainer(bst)
shap_values = explainer.shap_values(X_test)  # shap_values will be a list of arrays
print(shap_values[0].shape)
print(X_train.iloc[0,:].shape)

# shap_values = array(shap_values)
# shap.plots.bar(shap_values[0], max_display=10)  # 展示所有特征
# 可视化SHAP值摘要
shap.summary_plot(shap_values, X_test,max_display=13)
shap.summary_plot(shap_values[0], X_train.iloc[0,:],plot_type="bar")
plt.show()
对于单个实例的可视化
# 生成瀑布图

# shap.plots.waterfall(explainer.expected_value[1], shap_values[1][0], X_test.iloc[0, :],show=True)
# # 添加标题和显示图表
# plt.title('Waterfall Plot for Instance #{} - Predicting Class {}'.format(instance_index, class_index))
# plt.show()
# shap.force_plot(explainer.expected_value[1], shap_values[1][:, 0], X_test.iloc[0, :],matplotlib=True)
#
# plt.show()
# print(shap_values[:,:,0].shape)
# print(X_test.shape)
# print(X_test[73])
# 一个简单的函数，用于绘制带有特征名称及类别名称的SHAP力导向图
def plot_force_with_names(predicted_class, predicted_class_name, instance_index, shap_values, instance, explainer):
    expected_value = explainer.expected_value[predicted_class]
    shap_value = shap_values[predicted_class][instance_index]
    shap.force_plot(expected_value, shap_value, instance, feature_names=instance.index, show=False, matplotlib=True)
    plt.title(f'Force plot for prediction: {predicted_class_name}')
    plt.show()


for index, instance in X_test.iterrows():
    predicted_class = bst.predict([instance])[0]
    predicted_class_name = y_test[predicted_class]
    plot_force_with_names(predicted_class, predicted_class_name, index, shap_values, instance, explainer)