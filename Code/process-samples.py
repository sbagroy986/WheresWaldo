from PIL import Image, ImageOps
import os
from os import listdir
from os.path import isfile, join
from ImageAugmenter import ImageAugmenter
from scipy import misc
import numpy as np

# directory=os.getcwd()+"/training/"
# files = [f for f in listdir(directory) if isfile(join(directory, f))]

file="10_15_4.jpg"
image = misc.imread("./training/positive/"+file)
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
# augmented_images = augmenter.augment_batch(np.array([image], dtype=np.uint8))
# for image in augmented_images:
# 	augmented_images.plot()
fig=augmenter.plot_image(image,name=file, nb_repeat=5)
# for f in fig:
# 	f.show()
# fig.savefig("Verew")