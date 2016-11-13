from PIL import Image, ImageOps
import os
from os import listdir
from os.path import isfile, join
from ImageAugmenter import ImageAugmenter
from scipy import misc
import numpy as np

directory=os.getcwd()+"/training/positive_generated/"
files = [f for f in listdir(directory) if isfile(join(directory, f))]

for file in files:
	if "jpg" not in file and "png" not in file:
		continue
	image = misc.imread("./training/positive_generated/"+file)
	# image=Image.open("./training/positive/"+file)
	# for i in range(0,21):
	# 	image.save(os.getcwd()+"/training/positive_expanded/"+file.strip(".jpg").strip(".png")+str(i)+".jpg")
	height = image.shape[0]
	width = image.shape[1]

	augmenter = ImageAugmenter(width, height, # width and height of the image (must be the same for all images in the batch)
	                           hflip=True,    # flip horizontally with 50% probability
	                           vflip=True,    # flip vertically with 50% probability
	                           scale_to_percent=1.3, # scale the image to 70%-130% of its original size
	                           scale_axis_equally=False, # allow the axis to be scaled unequally (e.g. x more than y)
	                           rotation_deg=25,    # rotate between -25 and +25 degrees
	                           shear_deg=10,       # shear between -10 and +10 degrees
	                           translation_x_px=5, # translate between -5 and +5 px on the x-axis
	                           translation_y_px=5  # translate between -5 and +5 px on the y-axis
	                           )
	fig=augmenter.plot_image(image,name=file, nb_repeat=20)
