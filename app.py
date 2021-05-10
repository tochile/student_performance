from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap
from sklearn.preprocessing import LabelEncoder
import pickle 
import pandas as pd



app = Flask(__name__) 
Bootstrap(app)

@app.route('/')
def index():
	return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
	
    if request.method == 'POST':
        raisedhands = request.form['raisedhands']
        visited_resource = request.form['visited_resource']
        announcement = request.form['announcement']
        discussion = request.form['discussion']
        biology = request.form['biology']
        chemistry = request.form['chemistry']
        english = request.form['english']
        french = request.form['french']
        geology = request.form['geology']
        history = request.form['history']
        IT = request.form['IT']
        maths = request.form['maths']
        crs = request.form['crs']
        science = request.form['science']
        computer = request.form['computer']
        semester = request.form['semester']
        absent = request.form['abs']
       
        from keras.models import load_model
        import numpy as np
        new_model = load_model("student_model.h5")
        label_encoder = LabelEncoder()
        data = [raisedhands, visited_resource,announcement, discussion, biology,chemistry, english,french,
                geology,history,IT,maths, crs,science,computer,semester,absent]

        data = np.array(label_encoder.fit_transform(data))
        data1 = data.reshape(1,-1)
		
	
        pred = new_model.predict_classes(data1)
    return render_template('home.html', prediction = pred)

	



if __name__=='__main__':
	app.run(debug=True)