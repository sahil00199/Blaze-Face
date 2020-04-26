import numpy as np
import pickle
import tensorflow as tf
import os
from tensorflow import keras
import cv2

############################################### DATA GENERATION #####################################################
class DataGenerator(keras.utils.Sequence):
	# taken from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
	def __init__(self, split, anchors, batchSize=32, shuffle=True, dataDir = '../data/processed/'):
		'Initialization'
		self.batchSize = batchSize
		self.shuffle = shuffle
		self.dataDir = dataDir
		self.anchors = anchors
		########################## Generate the random data
		annotations = pickle.load(open(os.path.join(self.dataDir, 'annotations.pkl'), 'rb'))
		images = pickle.load(open(os.path.join(self.dataDir, 'images.pkl'), 'rb'))
		self.input, self.boundingBoxes = [], []
		for imageNumber, (imageName, annotation) in enumerate(annotations[split].items()):
			annotation[2] += annotation[0]
			annotation[3] += annotation[1]
			self.input.append(images[imageName])
			self.boundingBoxes.append(annotation)
			if split == 'train':
				if imageNumber == 5000: break
			else:
				if imageNumber == 1000: break
			################################## DEBUG ################################
			# if split == 'train':
			# 	groundTruth = [int(x + 0.5) for x in annotation]
			# 	print(imageName, groundTruth)
			# 	image = images[imageName]
			# 	# gt = cv2.rectangle(image, groundTruth[:2], groundTruth[2:])#, color = (255, 0, 0), thickness = 3)
			# 	image = cv2.rectangle(image, (groundTruth[0], groundTruth[1]), (groundTruth[0], groundTruth[1]),(255, 0, 0), thickness = 1)
			# 	cv2.imwrite('../data/predictions/debug_' + str(imageNumber) + '.jpg', image)
			###########################################################################

		self.numSamples = len(self.input)
		self.input = np.array(self.input)
		self.input = (self.input - 127.5 ) / 127.5
		# boundingBox has 4 entries - x1, y1, x2, y2: I randomly choose x1, y1 \in [0, 128] and x2, y2 \in [128, 256]
		self.boundingBoxes = np.array(self.boundingBoxes)
		self.classLabels = np.asarray([self.getLabels(i, boundingBox) for i, boundingBox in enumerate(self.boundingBoxes)])
		self.output = np.asarray([self.convertBoundsToCentreSide(boundingBox, labels) for boundingBox, labels in zip(self.boundingBoxes, self.classLabels)])
		print(split, self.input.shape[0])
		##########################
		self.on_epoch_end()

	def convertBoundsToCentreSide(self, boundingBox, labels):
		assert boundingBox[2] >= boundingBox[0], boundingBox
		assert boundingBox[3] >= boundingBox[1], boundingBox
		w = boundingBox[2] - boundingBox[0]
		h = boundingBox[3] - boundingBox[1]
		cx = (boundingBox[0] + boundingBox[2]) / 2.0
		cy = (boundingBox[1] + boundingBox[3]) / 2.0
		return np.array([[label, cx, cy, w, h] for label in labels])

	def convertCentreSideToBounds(self, centreSide):
		x1 = centreSide[0] - (centreSide[2]) / 2.0
		x2 = centreSide[0] + (centreSide[2]) / 2.0
		y1 = centreSide[1] - (centreSide[3]) / 2.0
		y2 = centreSide[1] + (centreSide[3]) / 2.0
		return np.asarray([x1, y1, x2, y2])

	def getLabels(self, index, boundingBox):
		global anchors
		if index % 500 == 0: print(index)
		return np.asarray([1 if self.iou(self.convertCentreSideToBounds(anchor), boundingBox) > 0.5 else 0 for anchor in self.anchors])

	def iou(self, a, b):
		# assert a[2] >= a[0], a
		# assert b[2] >= b[0], b
		# assert a[3] >= a[1], a
		# assert b[3] >= b[1], b
		intersect = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
		if (intersect[2] > intersect[0]) and (intersect[3] > intersect[1]):
			areaIntersection = (intersect[2] - intersect[0]) * (intersect[3] - intersect[1])
		else:
			areaIntersection = 0
		areaA = (a[2] - a[0]) * (a[3] - a[1])
		areaB = (b[2] - b[0]) * (b[3] - b[1])
		areaUnion = areaA + areaB - areaIntersection
		assert areaIntersection >= -0.0001
		assert areaUnion >= -0.0001
		return areaIntersection / areaUnion

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(self.numSamples / self.batchSize))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indices of the batch
		indices = self.indices[index*self.batchSize:(index+1)*self.batchSize]
		X = np.asarray([self.input[index] for index in indices])
		y = [np.asarray([self.classLabels[index] for index in indices]), np.asarray([self.output[index] for index in indices])]
		return X, y

	def on_epoch_end(self):
		'Updates indices after each epoch'
		self.indices = np.arange(self.numSamples)
		if self.shuffle == True:
			np.random.shuffle(self.indices)
#######################################################################################################

if __name__ == "__main__":
	DataGenerator('train')