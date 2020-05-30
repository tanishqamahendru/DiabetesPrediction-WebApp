#https://www.youtube.com/watch?v=02eZFXALcl4&t=186s

from flask import Flask, render_template, request, url_for
import pickle

app = Flask(__name__)

#open a file, where you stored the pickle data
file = open('model', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/diabetes')
def knowMore():
	return render_template('diabetes.html')


@app.route('/test', methods=["GET", "POST"])
def test():
	#code for inference
	if request.method == "POST":
		myDict = request.form
		pregnancies = int(myDict['pregnancies'])
		glucose = float(myDict['glucose'])
		insulin = float(myDict['insulin'])
		bmi = float(myDict['bmi'])
		age = int(myDict['age'])
		
		inputFeatures = [pregnancies, glucose, insulin, bmi, age]
		infProb = clf.predict([inputFeatures])[0]
		return render_template('show.html', inf=infProb)
	return render_template('index.html')
	#return 'Hello, World! ' + str(infProb)


if __name__ == "__main__":
	app.run(debug=True)