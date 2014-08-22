#Extract pixel locations that satisfy a certain THRESHOLD
#Author: Shane Barratt

from PIL import Image
import IPython as ipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial as ss

def read_image():
	#open image using PIL
	im = Image.open("source.jpg")

	pixels = np.zeros((im.size[0],im.size[1],3))

	#extract pixels from image
	for x in range(im.size[0]):
		for y in range(im.size[1]):
			pixels[x,y] = im.getpixel((x,y))

	return pixels

class Point:
	def __init__(self, x, y):
		self.x = x
		self.y = y
	def __hash__(self):
		return ( self.y << 16 ) ^ self.x;

	def __eq__(self, pt):
		return self.x == pt.x and self.y == pt.y


def down_sampling(pts,square_size):
	#get min/max x/y
	x_min,y_min= (int)(pts.min(axis=0)[0]),(int)(pts.min(axis=0)[1])
	x_max,y_max= (int)(pts.max(axis=0)[0]),(int)(pts.max(axis=0)[1])

	final_pts = np.zeros((0,2)) #initialize final_pts

	#initialize KDTree for efficient kNN
	pts_list = zip(pts[:,0],pts[:,1])
	tree = ss.KDTree(pts_list)

	x_dict = {}
	y_dict = {}
	ct=0

	for x in range(x_min,x_max+1):
		x_dict[x] = ct/square_size
		ct += 1
	ct=0
	for y in range(y_min,y_max+1):
		y_dict[y] = ct/square_size
		ct += 1

	box_ind_to_pts = {}

	for pt in pts:
		x,y = pt[0],pt[1]
		x_ind = x_dict[x]
		y_ind = y_dict[y]
		poi = Point(x_ind,y_ind)
		if poi in box_ind_to_pts.keys():
			box_ind_to_pts[poi].append((x,y))
		else:
			box_ind_to_pts[poi] = [(x,y)]

	for x_ind in set(x_dict.values()):
		for y_ind in set(y_dict.values()):
			if Point(x_ind,y_ind) in box_ind_to_pts.keys() and box_ind_to_pts[Point(x_ind,y_ind)] is not None:
				x_avg,y_avg = average_points(box_ind_to_pts[Point(x_ind,y_ind)])
				a,ind = tree.query((x_avg,y_avg),k=1)
				final_pts = np.r_[final_pts, np.array([[pts_list[ind][0],pts_list[ind][1]]])]

	"""
	for x in range(x_min,x_max,square_size):
		for y in range(y_min,y_max,square_size):
			ct_pts = 0
			pts_in_sq = []
			for pt in pts:
				#ipy.embed()
				x_1,y_1 = pt[0],pt[1]
				if x_1 in range((int)(x),(int)(x+square_size)) and y_1 in range((int)(y),(int)(y+square_size)):
					ct_pts += 1
					pts_in_sq.append((x_1,y_1))
			if ct_pts != 0:
				x_avg,y_avg = average_points(pts_in_sq)
				a,ind = tree.query((x_avg,y_avg),k=1)
				final_pts = np.r_[final_pts, np.array([[pts_list[ind][0],pts_list[ind][1]]])]
	"""
	return final_pts

def average_points(pts):
	i = len(pts)
	total_x,total_y=0,0
	for pt in pts:
		total_x += pt[0]
		total_y += pt[1]
	return (total_x/i,total_y/i)

def visualize(pts):
	plt.scatter(pts[:,0],-pts[:,1])
	plt.show()

def filter_points(pixels):
	#convert this into an np.array assuming meeting a certain threshold
	obj = np.zeros((0,2))
	for x in range(pixels.shape[0]):
		for y in range(pixels.shape[1]):
			if pixels[x,y][0] < 90: #THRESHOLD: red value < 70
				obj = np.r_[obj,np.array([[x,y]])]
	return obj

def main():
	pixels = read_image()
	drawing = filter_points(pixels)
	pts_ds = down_sampling(drawing,7)
	visualize(pts_ds)
	ipy.embed()

if __name__ == "__main__":
	main()