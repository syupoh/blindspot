# Imports
import random, cv2, os, sys, shutil
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
from tensorflow import keras

class image_clustering:

	def __init__(self, folder_path="/Data/jhshin/jfd_dataset/d1.15/", n_clusters=26):
		pass


if __name__ == "__main__":

	temp = image_clustering()
	temp.load
	data_path = "/Data/jhshin/jfd_dataset/d1.15/esophagus/" # path of the folder that contains the images to be considered for the clustering (The folder must contain only image files)

