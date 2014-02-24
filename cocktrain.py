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

def main():
	train_filename = "train.csv"
	with open(train_filename) as csvfile:
		rows = []
		reader = csv.reader(csvfile)
		for line in reader:
			rows.append(line)
	print "%d rows in %s" % (len(rows), train_filename)
	data = np.array(rows)
	headers = data[0,:]
	data = data[1:,:]

	split_row = 700
	label_train = data[:split_row,1]
	label_test = data[split_row:,1]

	# indices, as in the train.csv file
	categorical_col_indices = [4,11]
	numerical_col_indices = [2,5,6,7]

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

	print feats

	feats_train = feats[:split_row,:]
	feats_test = feats[split_row:,:]

	clf = svm.SVC()
	clf.fit(feats_train,label_train)
	print feats_test.shape
	print label_test.shape
	results = clf.score(feats_test,label_test)
	print results
	

if __name__=='__main__':
	main()

