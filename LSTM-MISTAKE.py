import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import MinMaxScaler  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import LSTM, Dense  
from sklearn.metrics import mean_squared_error  
import matplotlib.pyplot as plt

# 加载数据  
df = pd.read_csv("600000_2005.csv", index_col=0, parse_dates=True)  
  
# 提取特征  
features = df[['open', 'close', 'high', 'low', 'volume']]  
target = df['close']  
  
# 数据预处理：标准化  
scaler = MinMaxScaler(feature_range=(0, 1))  
scaled_features = scaler.fit_transform(features)  
scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))  
# 创建数据集（构建输入序列和目标）  
def create_dataset(dataset, look_back=1):  
    X, Y = [], []  
    for i in range(len(dataset) - look_back - 1):  
        a = dataset[i:(i + look_back), :]  
        X.append(a)  
        Y.append(dataset[i + look_back, 0])  
    return np.array(X), np.array(Y)  
  
look_back = 5  # 使用过去5天的数据预测  
X, y = create_dataset(scaled_features, look_back)  
  
# 将输入数据reshape为[samples, time steps, features]  
X = np.reshape(X, (X.shape[0], X.shape[1], 5))  
  
# 划分训练集和测试集  
train_size = int(len(X) * 0.7)  
test_size = len(X) - train_size  
train_X, test_X, train_y, test_y = X[:train_size], X[train_size:], y[:train_size], y[train_size:]  
  
# 构建LSTM模型  
model = Sequential()  
model.add(LSTM(50, activation='relu', input_shape=(look_back, 5)))  
model.add(Dense(1))  
  
model.compile(optimizer='adam', loss='mse')  
  
# 训练模型  
model.fit(train_X, train_y, epochs=200, batch_size=32, verbose=1)  
  
# 预测  
train_predict = model.predict(train_X)  
test_predict = model.predict(test_X)  
  
# 反标准化  
train_predict = scaler.inverse_transform(train_predict)  
train_y = scaler.inverse_transform(train_y.reshape(-1, 1))  
test_predict = scaler.inverse_transform(test_predict)  
test_y = scaler.inverse_transform(test_y.reshape(-1, 1))  
  
# 计算训练集和测试集的误差  
train_mse = mean_squared_error(train_y, train_predict)  
test_mse = mean_squared_error(test_y, test_predict)  
  
print(f'Train MSE: {train_mse:.4f}')  
print(f'Test MSE: {test_mse:.4f}')  
  
# 可视化结果（可选）
plt.plot(train_y, label='True Data')  
plt.plot(train_predict, label='Prediction')  
plt.title('Training Data')  
plt.xlabel('Time')  
plt.ylabel('Close Price')  
plt.legend()  
plt.show()  
  
plt.plot(test_y, label='True Data')  
plt.plot(test_predict, label='Prediction')  
plt.title('Test Data')  
plt.xlabel('Time')  
plt.ylabel('Close Price')  
plt.legend()  
plt.show()