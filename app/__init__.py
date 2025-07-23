from flask import Flask
app = Flask(__name__)
# 导入 routes 中定义的视图函数
from .routes import *