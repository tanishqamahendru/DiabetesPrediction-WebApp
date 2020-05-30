import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

if __name__ == "__main__":
	#read the data
	df = pd.read_csv('diabetes.csv', header=None)
	df.columns = ["pregnancies", "glucose", "diastolic", "triceps","insulin","bmi","dpf","age","diabetes"]
	df = df.drop(["diastolic","triceps","dpf"], axis=1)
	df.glucose.replace(0, np.nan, inplace=True)
	df.bmi.replace(0, np.nan, inplace=True)
	df.age.replace(0, np.nan, inplace=True)
	df.insulin.replace(0, np.mean(df.insulin), inplace=True)
	df = df.dropna()

	X_train = df[["pregnancies", "glucose", "insulin","bmi", "age"]].to_numpy()
	Y_train = df[["diabetes"]].to_numpy().reshape(752,)
	clf = LogisticRegression()
	clf.fit(X_train, Y_train)

	#open a file, where you want to store the data
	file = open('model', 'wb')      #here model.pkl is not working whereas model.pkl is working fine in jupyter, so here using name without .pkl

	#dump information to that file
	pickle.dump(clf, file)
	file.close()
