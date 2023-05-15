from flask import Flask,request,render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model_regression = pickle.load(open('models/Model_regression_height_weight.pkl','rb'))
model_scaled = pickle.load(open('models/Model_scaler_height_weight.pkl','rb'))


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/home',methods=['GET','POST'])
def predict_datapoint():
    
    if request.method == "POST":
        Weight = float(request.form.get('Weight'))
        
        ready_dataset = model_scaled.transform([[Weight]])
        result = model_regression.predict(ready_dataset)
        return render_template("home.html",result=result[0])
        
    else:
        return render_template('home.html')
        
if __name__=="__main__":
    app.run(host="0.0.0.0")