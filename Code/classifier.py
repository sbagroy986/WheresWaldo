from sklearn.cross_validation import KFold
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
	neg=neg[:len(pos)]
	return pos,neg

def kfold_split(positive,negative,k=5):
	y_pos=[]
	y_neg=[]
	y=[]
	for i in positive:
		y_pos.append(1)
	for i in negative:
		y_neg.append(0)


pos,neg=get_data()
print "Number of positive samples: ",len(pos)
print "Number of negative samples: ",len(neg)

