import random
from random import shuffle
from manipulate_image import split_image,merge_image
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,precision_score,recall_score
from sklearn.linear_model import LogisticRegression
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np

max_hist=1024
def get_hog(image):
	global max_hist
	i = Image.open(image)
	# print len(i.histogram())
	return i.histogram()[:768]

def get_data():
	directory=os.getcwd()+"/training/positive_expanded/"
	files = [f for f in listdir(directory) if isfile(join(directory, f))]
	pos=[]
	for file in files:
		if "jpg" not in file and "png" not in file:
			continue
		pos.append(get_hog(directory+"/"+file))

	directory=os.getcwd()+"/training/negative_generated/"
	files = [f for f in listdir(directory) if isfile(join(directory, f))]
	no=(int(0.8*len(pos)*4+0.2*len(pos)))+5
	print no
	sample=random.sample(range(0, len(files)), no)
	neg=[]
	for s in sample:
		file=files[s]
		if "jpg" not in file and "png" not in file:
			continue
		neg.append(get_hog(directory+"/"+file))
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
	return cross_val_data

def classifier(cross_val_data):
	acc=[]
	for i in range(1,6):
		# model=LinearSVC()
		model=LogisticRegression()
		model.fit(cross_val_data[i]['train_features'],cross_val_data[i]['train_labels'])
		preds=model.predict(cross_val_data[i]['test_features'])
		y=cross_val_data[i]['test_labels']
		print "Fold ",i
		print accuracy_score(y,preds)
		print precision_score(y,preds)
		print recall_score(y,preds)
		acc.append(accuracy_score(y,preds))
		preds=model.predict(cross_val_data[i]['train_features'])
		y=cross_val_data[i]['train_labels']
		print 
		print accuracy_score(y,preds)
		print 
		print
	print "Mean accuracy: ",np.mean(acc)
	train=cross_val_data[1]['train_features']+cross_val_data[1]['test_features']
	labels=cross_val_data[1]['train_labels']+cross_val_data[1]['test_labels']
	# model=LinearSVC()
	model=LogisticRegression()
	model.fit(train,labels)
	return model

def test_on_image(image,model):
	split_image(image,64,64)
	merge_image(image,model,64,64)

def test(image):
	image=os.getcwd()+"/images/"+image
	image=Image.open(image)
	hist=image.histogram()
	return hist

pos,neg=get_data()
print "Number of positive samples: ",len(pos)
print "Number of negative samples: ",len(neg)
cross_val_data=kfold_split(pos,neg)
model=classifier(cross_val_data)
test_on_image("5.jpg",model)

