import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras

############################################### DATA GENERATION #####################################################
class DataGenerator(keras.utils.Sequence):
	# taken from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
	def __init__(self, numSamples, anchors, batchSize=32, shuffle=True):
		'Initialization'
		self.anchors = anchors
		self.numSamples = numSamples
		self.batchSize = batchSize
		self.shuffle = shuffle
		########################## Generate the random data
		# input is 3x128 for every image
		self.input = np.random.randint(0, 256, (numSamples, 128, 128, 3))
		self.input = (self.input - 127.5 ) / 127.5
		# boundingBox has 4 entries - x1, y1, x2, y2: I randomly choose x1, y1 \in [0, 128] and x2, y2 \in [128, 256]
		self.boundingBoxes = np.zeros((numSamples, 4))
		self.boundingBoxes[:, :2] = np.random.randint(0, 64, (numSamples, 2))
		self.boundingBoxes[:, 2:] = np.random.randint(64, 128, (numSamples, 2))
		self.classLabels = [self.getLabels(boundingBox) for boundingBox in self.boundingBoxes]
		self.output = [self.convertBoundsToCentreSide(boundingBox, labels) for boundingBox, labels in zip(self.boundingBoxes, self.classLabels)]
		##########################
		self.on_epoch_end()

	def convertBoundsToCentreSide(self, boundingBox, labels):
		assert boundingBox[2] > boundingBox[0], boundingBox
		assert boundingBox[3] > boundingBox[1], boundingBox
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

	def getLabels(self, boundingBox):
		return np.asarray([1 if self.iou(self.convertCentreSideToBounds(anchor), boundingBox) > 0.5 else 0 for anchor in self.anchors])

	def iou(self, a, b):
		assert a[2] > a[0], a
		assert b[2] > b[0], b
		assert a[3] > a[1], a
		assert b[3] > b[1], b
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
