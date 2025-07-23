import tushare as ts
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from datetime import datetime
from keras.models import load_model

def predict_price(stock_code):
    # 获取当前的日期和时间
    current_time = datetime.now()
    # 从当前时间中提取年份
    year = current_time.year
    # 从tushare中获取股票数据,并将数据转化成csv表格存储
    data = ts.get_k_data(str(stock_code),start="2005-05-05")
    data.to_csv("stock_data.csv",index=False)
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

    # # 构建LSTM模型
    # model = Sequential()
    # model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, X.shape[2])))
    # model.add(LSTM(units=50))
    # model.add(Dense(X.shape[2]))  # 输出层单元数应与特征数相同
    # model.compile(optimizer=Adam(), loss='mean_squared_error')

    # # 训练模型
    # model.fit(X, Y, epochs=100, batch_size=32, verbose=2)

    # 加载模型  
    model_load_path = 'lstm_stock_model.h5'  # 使用与保存时相同的路径和文件名  
    model = load_model(model_load_path)
    # 使用模型进行预测
    last_sequence = X[-1].reshape(1, look_back, X.shape[2])
    next_day_prediction_scaled = model.predict(last_sequence)

    # 反归一化预测结果
    next_day_prediction = scaler.inverse_transform(next_day_prediction_scaled)

    # 取出'close'列的预测值
    predicted_close_price = next_day_prediction[-1,1]  # 'close'价格在第二列
    return float(predicted_close_price)