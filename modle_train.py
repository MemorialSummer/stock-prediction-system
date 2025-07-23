import tushare as ts
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime

# 获取当前的日期和时间
current_time = datetime.now()
# 从当前时间中提取年份
year = current_time.year
# 从tushare中获取股票数据,并将数据转化成csv表格存储
data = ts.get_k_data(str(600000),start="2005-05-05")
data.to_csv("stock_data.csv",index=False)
# 数据标准化
data = pd.read_csv('stock_data.csv', index_col='date', parse_dates=['date'])
# 选择 'open', 'close', 'high', 'low', 'volume' 五列作为特征
features = data[['open', 'close', 'high', 'low', 'volume']]
# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)
# 创建序列数据
def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data) - look_back - 1):
        a = data[i:(i + look_back), :]
        X.append(a)
        Y.append(data[i + look_back, :])
    return np.array(X), np.array(Y)
look_back = 3
X, Y = create_dataset(scaled_features, look_back)
# 重塑数据以适配LSTM [samples, time_steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))
# 构建LSTM模型
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, X.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(X.shape[2]))  # 输出层单元数应与特征数相同
model.compile(optimizer=Adam(), loss='mean_squared_error')
# 训练模型
model.fit(X, Y, epochs=100, batch_size=32, verbose=2)
# 保存模型  
model_save_path = 'lstm_stock_model.h5'  # 指定模型保存的路径和文件名  
model.save(model_save_path)  # 保存模型