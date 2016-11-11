from random import shuffle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np

def get_hog(image):
	i = Image.open(image)
	return np.array(i.histogram())


def get_data():
	directory=os.getcwd()+"/training/positive_expanded/"
	files = [f for f in listdir(directory) if isfile(join(directory, f))]
	pos=[]
	for file in files:
		if "jpg" not in file and "png" not in file:
			continue
		pos.append(get_hog(directory+"/"+file))

	directory=os.getcwd()+"/training/negative/"
	files = [f for f in listdir(directory) if isfile(join(directory, f))]
	neg=[]
	for file in files:
		if "jpg" not in file and "png" not in file:
			continue
		neg.append(get_hog(directory+"/"+file))
	shuffle(neg)
	neg=neg[:(int(0.8*len(pos)*4+0.2*len(pos)))]
	return pos,neg

def kfold_split(positive,negative,k=5):
	cross_val_data={}
	shuffle(positive)
	shuffle(negative)
	step_pos=int(0.2*len(positive))
	step_neg=int(0.2*len(negative))
	for i in range(1,6):
		cross_val_data[i]={}
		cross_val_data[i]['train_features']=[]
		cross_val_data[i]['train_labels']=[]
		cross_val_data[i]['test_features']=[]
		cross_val_data[i]['test_labels']=[]
		test=[]

		if i!=5:
			for t in range((i-1)*step_pos,i*step_pos):
				cross_val_data[i]['test_features'].append(list(positive[t]))
				cross_val_data[i]['test_labels'].append(1)
				test.append(t)
			for t in range((i-1)*step_pos,i*step_pos):
				cross_val_data[i]['test_features'].append(list(negative[t]))
				cross_val_data[i]['test_labels'].append(0)
				test.append(t)
		else:
			for t in range((i-1)*step_pos,len(positive)):
				cross_val_data[i]['test_features'].append(list(positive[t]))
				cross_val_data[i]['test_labels'].append(1)
				test.append(t)
			for t in range((i-1)*step_pos,len(positive)):
				cross_val_data[i]['test_features'].append(list(negative[t]))
				cross_val_data[i]['test_labels'].append(0)
				test.append(t)

		for t in range(len(positive)):
			if t not in test:
				cross_val_data[i]['train_features'].append(list(positive[t]))
				cross_val_data[i]['train_labels'].append(1)
		for t in range(len(negative)):
			if t not in test:
				cross_val_data[i]['train_features'].append(list(negative[t]))
				cross_val_data[i]['train_labels'].append(0)			
		# print "Fold: ",i
		# print "Train: ",len(cross_val_data[i]['train_features'])
		# print "Test: ",len(cross_val_data[i]['test_features'])		
	return cross_val_data

def classifier(cross_val_data):
	for i in range(1,6):
		model=LinearSVC()
		print cross_val_data[i]['test_labels']
		model.fit(cross_val_data[i]['train_features'],cross_val_data[i]['train_labels'])
		preds=model.predict(cross_val_data[i]['test_features'])
		y=cross_val_data[i]['test_labels']
		print "Fold ",i
		print accuracy_score(y,preds)

pos,neg=get_data()
print "Number of positive samples: ",len(pos)
print "Number of negative samples: ",len(neg)
cross_val_data=kfold_split(pos,neg)
classifier(cross_val_data)
