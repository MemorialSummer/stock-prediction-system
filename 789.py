import pandas as pd  
from statsmodels.tsa.arima.model import ARIMA  
from sklearn.metrics import mean_squared_error  
import matplotlib.pyplot as plt  
  
# 加载数据  
df = pd.read_csv("600000_2005.csv", index_col=0, parse_dates=True)  
  
# 提取close价格  
close_prices = df['close']  
  
# 划分训练集和测试集  
train_size = int(len(close_prices) * 0.7)  
train, test = close_prices[0:train_size], close_prices[train_size:]  
  
# 拟合ARIMA模型  
# 注意：这里p, d, q参数是ARIMA模型的阶数，需要手动选择或进行模型选择  
# 我们可以使用如auto_arima之类的工具来自动选择最优参数  
p = 1  
d = 1  
q = 0  
model = ARIMA(train, order=(p, d, q))  
model_fit = model.fit(disp=0)  
  
# 预测  
train_predict = model_fit.predict(start=0, end=len(train)-1)  
test_start = len(train)  
test_end = len(test)+len(train)-1  
test_predict = model_fit.predict(start=test_start, end=test_end, typ='levels')  
  
# 计算误差  
train_mse = mean_squared_error(train, train_predict)  
test_mse = mean_squared_error(test, test_predict[test_start-test_end:])  # 注意切片以匹配测试集长度  
  
print(f'Train MSE: {train_mse:.4f}')  
print(f'Test MSE: {test_mse:.4f}')  
  
# 可视化结果  
plt.plot(train, label='Train Data')  
plt.plot(train_predict, label='Train Prediction')  
plt.title('ARIMA Model - Training Data')  
plt.xlabel('Time')  
plt.ylabel('Close Price')  
plt.legend()  
plt.show()  
  
plt.plot(test, label='Test Data')  
plt.plot(range(test_start, test_end+1), test_predict[test_start-test_end:], label='Test Prediction')  
plt.title('ARIMA Model - Test Data')  
plt.xlabel('Time')  
plt.ylabel('Close Price')  
plt.legend()  
plt.show()