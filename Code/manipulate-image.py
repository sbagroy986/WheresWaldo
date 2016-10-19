from PIL import Image, ImageOps
import os
from os import listdir
from os.path import isfile, join
# import Image

def split_image(height,width):
	directory=os.getcwd()+"/images/"
	files = [f for f in listdir(directory) if isfile(join(directory, f))]
	for file in files:
		if file.strip(".png").strip(".jpg")!="19":
			continue
		segment_dir=os.getcwd()+"/segments/"
		if "jpg" not in file and "png" not in file:
			continue
		if not os.path.exists(segment_dir+file.strip(".jpg").strip(".png")):
			os.makedirs(segment_dir+file.strip(".jpg").strip(".png"))
		image=Image.open(directory+file)
		image_width, image_height = image.size
		for i in range(0,image_height,height):
			for j in range(0,image_width,width):
				dim=(j,i,j+width,i+height)
				patch=image.crop(dim)
				path=os.path.join(os.getcwd()+'/segments/'+file.strip(".jpg").strip(".png")+"/","IMG_%s_%s.png" % (i,j))
				patch.save(path)

def merge_image(height,width):
	directory=os.getcwd()+"/images/"
	fs = [f for f in listdir(directory) if isfile(join(directory, f))]
	for fn in fs:
		# fn=fn.strip(".png").strip(".jpg")
		path=os.getcwd()+'/segments/'+fn.strip(".png").strip(".jpg")
		files = [f for f in listdir(path) if isfile(join(path, f))]
		max_i=0
		max_j=0
		for i in files:
			if "png" not in i and"jpg" not in i:
				continue
			name=i.split("_")
			max_j=max(max_j,int(name[2].strip(".png")))
			max_i=max(max_i,int(name[1]))
		image=Image.new("RGB", (max_j,max_i))
		for file in files:
			if "png" not in file and "jpg" not in file:
				continue
			temp_image=Image.open(path+"/"+file)
			name=file.split("_")
			i=int(name[1])
			j=int(name[2].strip(".png"))
			if i==640 and j==320 and fn.strip(".png").strip(".jpg")=="19":
				size=61,61
				temp_image.thumbnail(size, Image.ANTIALIAS)
				temp_image.save(path+"/"+file)
				temp_image=Image.open(path+"/"+file)
				ImageOps.expand(temp_image,border=(2,2),fill='red').save(path+"/"+file)
			temp_image=Image.open(path+"/"+file)
			image.paste(temp_image,(j,i))
		image.save(os.getcwd()+"/merged/"+fn)

split_image(64,64)
merge_image(64,64)