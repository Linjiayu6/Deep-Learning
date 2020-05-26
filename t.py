from flask import Flask, render_template, request
# from flask_cors import *

app = Flask(__name__)
# r'/*' 是通配符，让本服务器所有的URL 都允许跨域请求
# CORS(app, resources=r'/*')

@app.route('/index/')
def index():
    return render_template('index.html')

@app.route('/', methods = ["GET", "POST"])
def post_data():
    name = request.args.get("name")
    age = request.args.get("age")

    # post请求是通过 flask.request.form('name')

    return name + age

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)
