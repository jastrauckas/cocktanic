import numpy as np
import scipy
from sklearn import svm
import csv



def main():
	with open("train.csv") as csvfile:
		rows = []
		reader = csv.reader(csvfile)
		for line in reader:
			print line
			rows.append(line)
	print len(rows)
	data = np.array(rows)
	print data.shape
	split_row = 700
	label_train = data[:split_row,1]
	feats_train = data[:split_row,2:]

	label_test = data[split_row:,1]
	feats_test = data[split_row:,2:]

	clf = svm.SVC()
	clf.fit(feats_train,label_train)
	results = clf.score(feats_test,label_test)
	print results
	

if __name__=='__main__':
	main()

