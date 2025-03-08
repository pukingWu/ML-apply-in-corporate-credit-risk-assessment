import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# 假设我们有一维序列数据和对应的9个类别的标签
# X 是形状为 (num_samples, sequence_length, num_features) 的数据
# y 是形状为 (num_samples,) 的标签数据

# 这里只是示例数据维度，你需要用实际的数据替换
file = r'C:\Users\吴振波\Desktop\mltest1.csv'
cred = pd.read_csv(file)
pd.set_option('future.no_silent_downcasting', True)
# 注意：您应该使用 inplace=True 或重新赋值来确保替换是永久性的
cred.replace({'企业信用等级': {'C': 6, 'CC': 7, 'CCC': 0, 'B': 1, 'BB': 2, 'BBB': 3, 'A': 4, 'AA': 5, 'AAA': 9}},
             inplace=True)
X = cred.drop(columns=['企业信用等级'])
y = cred['企业信用等级'].astype(int)

# num_samples = 1000
# sequence_length = 100
# num_features = 1
# num_classes = 9

# 随机生成示例数据
# X = np.random.rand(num_samples, sequence_length, num_features)
# y = np.random.randint(0, num_classes, size=(num_samples,))

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 对标签进行独热编码
y_train = tf.keras.utils.to_categorical(y_train, num_classes=6)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=6)

# 构建1D CNN模型
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(3,13,64)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(6, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

# 特征提取：和之前类似，我们移除顶层，创建一个新的模型用于特征提取
feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)

# 提取特征
features_train = feature_extractor.predict(X_train)

print(features_train)
# 现在你可以使用这些特征进行进一步的分析或训练其他模型