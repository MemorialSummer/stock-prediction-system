from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from model.lstm_model import predict_price as predict_price_lstm
from model.arima_model import predict_price as predict_price_arima
# from model.rnn_model import predict_price as predict_price_rnn
from datetime import datetime,timedelta
import csv  
import sqlite3 
import tushare as ts
from PIL import Image
import mplfinance
import base64
import pandas as pd
#模拟数据库提供登录用户名和密码
users_db = {
    'sum': '123'
}
#检查用户名密码是否相符合
def check_user(username, password):
    if username in users_db and users_db[username] == password:
        return True
    else:
        return False


app = Flask(__name__)


# 主页面,登录页面
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        cid = request.form['cid']
        password = request.form['password']
        # 函数check_user，它返回一个布尔值表示用户是否存在
        user_exists = check_user(cid, password)
        
        if user_exists:
            # 如果用户存在，重定向到另一个页面optional
            return redirect(url_for('optional'))
        else:
            # 如果用户不存在，重新渲染登录页面，并显示错误消息
            return render_template('login.html', error="Invalid username or password")
    else:
        # 如果是GET请求，渲染登录表单
        return render_template('login.html')


# 自选股票页面
@app.route('/optional')
def optional():
    return render_template('optional.html')


#其他股票信息查询页面
@app.route('/information')
def information():
    return render_template('information.html')

@app.route('/prediction', methods=['GET'])
def index():
    return render_template('predict.html')

#股票预测界面
@app.route('/predict',methods=['POST'])
def get_prediction():
    data = request.json
    stock_code = data.get('stock_code')
    model = data.get('model', 'lstm_model')  # 默认使用LSTM模型

    if not stock_code:
        return jsonify({'error': 'Stock code is required'}), 400

    try:
        if model == 'lstm_model':
            prediction = predict_price_lstm(stock_code)
        elif model == 'arima_model':
            prediction = predict_price_arima(stock_code)
        # elif model == 'rnn_model':
        #     prediction = predict_price_rnn(stock_code)
        else:
            return jsonify({'error': 'Model not supported'}), 400

        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    


@app.route('/stock_info')
def stock_info():
    # 获取表单提交的股票代码
    # 获取当前的日期和时间
    current_time = datetime.now()
    sixty_days_ago = current_time - timedelta(days=60)

    stock_code = request.args.get('stock_code', type=str)
    if not stock_code:
        return "未提供股票代码。"
    
    try:
        # 获取近60天的股票信息
        df = ts.get_k_data(stock_code,start=sixty_days_ago.strftime('%Y-%m-%d'))
        df.to_csv("information_search.csv",index=False)
        # 存储到数据库
        
        # 连接到SQLite数据库（如果不存在，则创建一个）  
        conn = sqlite3.connect('D:/newDatabase.db')  
        c = conn.cursor()  
        
        # 创建一个表（如果表已经存在，你需要修改或删除这个步骤）  
        c.execute('''CREATE TABLE IF NOT EXISTS Stock_'''+str(stock_code)+''' (Date CHAR(20),Open FLOAT,Close FLOAT,High FLOAT,Low FLOAT,Volume FLOAT,Code INT)''')  
        c.execute('''DELETE FROM Stock_'''+str(stock_code))  
        # 读取CSV文件  
        with open('information_search.csv', 'r') as csvfile:  
            csvreader = csv.reader(csvfile)  
            
            # 跳过标题行（如果CSV文件有标题行）  
            next(csvreader)  
            
            # 遍历CSV文件的每一行，并将其插入到SQLite数据库的表中  
            for row in csvreader:  
                i = 1
                # 假设CSV文件的列与数据库表的列相匹配  
                c.execute("INSERT INTO Stock_"+str(stock_code)+" (Date,Open,Close,High,Low,Volume,Code) VALUES (?, ?, ?, ?, ?, ?, ?)", (row[0], row[1], row[2], row[3], row[4], row[5], row[6]))  
                i = i+1
        # 提交事务并关闭连接  
        conn.commit()  
        conn.close()
        # 将DataFrame转换为HTML表格
        table_html = df.to_html(classes='table table-striped table-hover')
        
        # 渲染模板，传递股票信息
        return render_template('information.html', stock_table=table_html)
    
    except Exception as e:
        return f"发生错误: {e}"


@app.route('/kline')
def kline():
    current_time = datetime.now()
    sixty_days_ago = current_time - timedelta(days=60)
    stock_code = request.args.get('stock_code', type=str)
    df = ts.get_k_data(stock_code,start=sixty_days_ago.strftime('%Y-%m-%d'))
    df.to_csv("stock_data.csv",index=False)
    df = pd.read_csv("stock_data.csv",index_col=0,parse_dates=[0])
    # 创建一个自定义的样式  
    style = mplfinance.make_mpf_style(base_mpf_style='charles', rc={'figure.figsize':(10, 7)})
    mplfinance.plot(df, type='candle', mav=(3, 6, 9), style=style, volume=True, savefig='app/static/output.png', show_nontrading=False)  
    # 将fig保存为图像，而不是显示它
    return render_template('kline.html')