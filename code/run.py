''' 

Main script that runs everything.

'''

import fileparse as docR
from sklearn import svm
from sklearn.externals import joblib
import pickle 
import sys
import numpy as np

if __name__=="__main__":
	if len(sys.argv) < 4:
		print "USAGE: $ python run.py [-n | -o] model_file data_file"

	if sys.argv[1] == "-n":
		docs_list = docR.get_list()
		classifier = svm.SVR()
		data = []
		target = []
		for doc in docs_list:
			data.append(doc.vector[:-1])
			target.append(doc.vector[-1])
		np_data = np.array(data)
		np_target = np.array(target)
		classifier.fit(np_data, np_target)
		joblib.dump(classifier, sys.argv[2])
		save_data = data
		save_data.append(target)
		string = pickle.dumps(save_data)
		ofp = open(sys.argv[3], 'w')
		ofp.write(string)
		ofp.close()
		print "Model and data dumped successfully."

	elif sys.argv[1] == "-o":
		classifier = joblib.load(sys.argv[2])
		saved_data = open(sys.argv[3], 'r')
		saved_data = saved_data.read()
		saved_data = pickle.loads(saved_data)
		data = np.array(saved_data[:-1])
		target = np.array(saved_data[-1])
		print "Model and data loaded successfully."
