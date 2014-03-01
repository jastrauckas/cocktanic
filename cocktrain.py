import numpy as np
import scipy
from sklearn import svm, cross_validation, tree
import csv
from sklearn.externals.six import StringIO
import pydot

'''
takes in a feature vector with categorical entries
returns a vector with an integer mapping of the features
'''
def mapCategoriesToInts(feat_vector):
	categories = list(set(feat_vector))
	for i,c in enumerate(feat_vector):
		feat_vector[i] = categories.index(c)
	return feat_vector

'''
takes in the header of a file and a matrix of data
returns a feature matrix
'''
def getFeatures(header,data):
	categorical_col_fields = ["Sex"] #,"Embarked"]	#[4,11]
	numerical_col_fields = ["Pclass"] #,"Age","SibSp","Parch"]	# [2,5,6,7]
	categorical_col_indices = map(lambda x: header.index(x), categorical_col_fields)
	numerical_col_indices = map(lambda x: header.index(x), numerical_col_fields)

	feats = data[:,numerical_col_indices]
	for r in range(feats.shape[0]):
		for c in range(feats.shape[1]):
			if feats[r,c] == '':
				feats[r,c] = -1

	num_feats = feats.shape[0]
	for col in categorical_col_indices:
		int_vector = mapCategoriesToInts(data[:,col])
		# reshape the vector to be able to concatenate it
		int_vector = int_vector.reshape((num_feats,1))
		feats = np.hstack((feats, int_vector))

	return feats

'''
takes in a filename
returns a 2D list with the contents of the file
'''
def readCSV(filename):
	with open(filename) as csvfile:
		rows = []
		reader = csv.reader(csvfile)
		for line in reader:
			rows.append(line)
	print "%d rows in %s" % (len(rows), filename)
	return rows


'''
takes in the training features and the training labels
prints out the performance accuracy
'''
def testOnTraining(X,y,classifier='tree'):
	n = X.shape[0]
	kf = cross_validation.KFold(n, n_folds=20)

	if classifier == 'svm':
		clf = svm.SVC(kernel="rbf")
	elif classifier == 'tree':
		clf = tree.DecisionTreeClassifier()
	accuracies = []
	for train_index, test_index in kf:
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		clf.fit(X_train,y_train)
		accuracies.append( clf.score(X_test,y_test) )


	ave_accuracy = sum(accuracies)/len(accuracies)
	print "%f performance accuracy" % ave_accuracy


'''
write out a file with the predicted labels of the test set
'''
def createPredictions(feats_train,label_train,feats_test,pid_col):
	clf = svm.SVC(kernel="rbf")
	clf.fit(feats_train,label_train)
	predictions = clf.predict(feats_test)
	filename = "cocktanic_submission.csv"
	with open(filename,"wb") as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(["PassengerId","Survived"])
		for pred,pid in zip(predictions,pid_col):
			writer.writerow([pid,0])
		

def main():
	# get a predicted accuracy
	train_filename = "train.csv"
	train_rows = readCSV(train_filename)

	data = np.array(train_rows)
	header = list(data[0,:])
	data = data[1:,:]

	label_train = data[:,1]

	feats = getFeatures(header,data)
	testOnTraining(feats,label_train)

	# generate predictions on test data
	test_filename = "test.csv"
	test_rows = readCSV(test_filename)

	data_test = np.array(test_rows)
	header_test = list(data_test[0,:])
	data_test = data_test[1:,:]

	pid_col = data_test[:,0]
	label_test = data_test[:,1]
	feats_test = getFeatures(header_test,data_test)
	
	createPredictions(feats,label_train,feats_test,pid_col)


if __name__=='__main__':
	main()

