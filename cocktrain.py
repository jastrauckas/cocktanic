import numpy as np
import scipy
from sklearn import svm
import csv



def main():
	with open("train.csv") as csvfile:
		rows = []
		reader = csv.reader(csvfile)
		for line in reader:
			#print line
			rows.append(line)
	print len(rows)
	data = np.array(rows)
	#print data.shape
	headers = data[0,:]
	data = data[1:,:]

	split_row = 700
	label_train = data[:split_row,1]
	label_test = data[split_row:,1]

	feats = data[:,2:]	
	# consolidate numeric columns and then remove the rest
	# I feel certain there is a more elegant way to do this
	feats[:,2] = feats[:,0]
	feats[:,6] = feats[:,7]
	feats = feats[:,2:6]
	for r in range(feats.shape[0]):
		for c in range(feats.shape[1]):
			if feats[r,c] == '':
				feats[r,c] = -1

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

