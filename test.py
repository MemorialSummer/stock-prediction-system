import matplotlib.pyplot as plt
import mplfinance
import pandas as pd
# import tushare as ts
# plt.plot([1, 2, 3, 4])
# plt.savefig('my_figure.png')  # 保存图形到文件
# plt.close()  # 关闭图形，释放资源
# stock_code = 600000
# df = ts.get_k_data(stock_code,start='2023-01-01', end='2023-12-31')
# df.to_csv("stock_data.csv",index=False)
df = pd.read_csv("information_search.csv",index_col=0,parse_dates=[0])

# 绘制K线图
# 使用mplfinance绘制K线图但不显示 
# mplfinance.plot(df, type='candle', mav=(3,6,9), volume=True, style='charles', savefig='output.png', show_nontrading=False)

# 创建一个自定义的样式  
style = mplfinance.make_mpf_style(base_mpf_style='charles', rc={'figure.figsize':(10, 7)})

# # 使用mpf.plot的return_fig参数来获取Figure和Axes对象  
mplfinance.plot(df, type='candle', mav=(3,6,9), volume=True, style=style, savefig='output.png', show_nontrading=False)
# 保存图表到文件  
# fig.savefig('output_custom.png')  