# 假设 df 是一个包含股票历史价格的 DataFrame
# 这里只是一个示例，你需要用你的 LSTM 模型来替换这部分代码
import tushare as ts
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime


def predict_price(stock_code):
    # 获取当前的日期和时间
    current_time = datetime.now()
    
    # 从当前时间中提取年份
    year = current_time.year

    # 从tushare中获取股票数据,并将数据转化成csv表格存储
    data = ts.get_k_data(str(stock_code),start="2005-05-05")
    data.to_csv("stock_data.csv",index=False)
    data = pd.read_csv("stock_data.csv",index_col=0,parse_dates=[0])

    # 数据重采样并选取训练集
    stock_week = data["close"].resample("W").mean()
    stock_train = stock_week["2005":str(year)].dropna()
    # print(stock_train)
    # stock_train.plot(figsize=(12,8))
    # plt.show()

    # 训练ARIMA模型
    model = ARIMA(stock_train,order=(2,0,2))
    result = model.fit()
    # print(result.summary())

    last_index = stock_train.index.max()
    # 预测未来
    prediction = result.predict(last_index,last_index)

    # plt.figure(figsize=(6,6))
    # plt.plot(pred)
    # plt.plot(stock_week.values)
    # plt.show()
    # prediction = result.predict(640,640,dynamic=True)['stock_price']
    return prediction[0]