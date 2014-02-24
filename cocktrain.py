import numpy as np
import scipy
from sklearn import svm
import csv


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
	categorical_col_fields = ["Sex","Embarked"]	#[4,11]
	numerical_col_fields = ["Pclass","Age","SibSp","Parch"]	# [2,5,6,7]
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

def main():
	train_filename = "train.csv"
	train_rows = readCSV(train_filename)

	data = np.array(train_rows)
	header = list(data[0,:])
	data = data[1:,:]

	split_row = 700
	label_train = data[:split_row,1]
	label_test = data[split_row:,1]

	feats = getFeatures(header,data)
	feats_train = feats[:split_row,:]
	feats_test = feats[split_row:,:]

	clf = svm.SVC(kernel="rbf")
	clf.fit(feats_train,label_train)
	print feats_test.shape
	print label_test.shape
	results = clf.score(feats_test,label_test)
	print results
	

if __name__=='__main__':
	main()

