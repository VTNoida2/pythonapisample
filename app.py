from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def index():    
    return render_template("index.html")

@app.route("/")
def hello():
    return "Hello, World!"	
	
if __name__ == '__main__':
    app.run(debug=True)
        