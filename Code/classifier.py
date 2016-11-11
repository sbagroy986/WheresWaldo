import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np

def get_hog(image):
	i = Image.open(image)
	return np.array(i.histogram())


def get_files():
	directory=os.getcwd()+"/training/negative/"
	files = [f for f in listdir(directory) if isfile(join(directory, f))]
	neg=[]
	for file in files:
		if "jpg" not in file and "png" not in file:
			continue
		neg.append(file)


	directory=os.getcwd()+"/training/positive_expanded/"
	files = [f for f in listdir(directory) if isfile(join(directory, f))]
	pos=[]
	for file in files:
		if "jpg" not in file and "png" not in file:
			continue
		pos.append(file)
	return pos,neg

pos,neg=get_files()
print "Number of positive samples: ",len(pos)
print "Number of negative samples: ",len(neg)
