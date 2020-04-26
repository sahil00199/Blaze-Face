import tensorflow as tf
from tensorflow import keras
import numpy as np
from network import BlazeNet, SSD
from dataset import DataGenerator
import cv2
import os

##########################################  LOSS ################################################
def smoothL1(groundTruth, predictions, globalMask):
	# compute smooth L1
	diff = predictions - groundTruth
	mask = tf.cast(tf.math.abs(diff) < 1, dtype=tf.float32)
	loss = ((diff ** 2) * 0.5) * mask + (tf.math.abs(diff) - 0.5) * (1.0 - mask)
	loss = tf.keras.backend.sum(loss * globalMask)
	return loss
	
def boundingBoxLoss(groundTruth, predictions):
	global anchors, alpha
	# First element (of groundTruth) is the class label - which is 1 if IOU is over 0.5. That is when the loss is to be applied
	mask = groundTruth[:, :, 0]
	# transform the ground truth
	gx = (groundTruth[:, :, 1] - anchors[:, 0]) / anchors[:, 2]
	gy = (groundTruth[:, :, 2] - anchors[:, 1]) / anchors[:, 3]
	gw = tf.math.log(groundTruth[:, :, 3] / anchors[:, 2])
	gh = tf.math.log(groundTruth[:, :, 4] / anchors[:, 3])

	loss = smoothL1(gx, predictions[:, :, 0], mask) +\
			smoothL1(gy, predictions[:, :, 1], mask) +\
			smoothL1(gw, predictions[:, :, 2], mask) +\
			smoothL1(gh, predictions[:, :, 3], mask)
	return loss * alpha


def smoothL1Debug(groundTruth, predictions, globalMask):
	# compute smooth L1
	diff = predictions - groundTruth
	mask = np.abs(diff) < 1
	loss = ((diff ** 2) * 0.5) * mask + (np.abs(diff) - 0.5) * (1.0 - mask)
	loss = np.sum(loss * globalMask)
	return loss
	
def boundingBoxLossDebug(groundTruth, predictions):
	global anchors, alpha
	# First element (of groundTruth) is the class label - which is 1 if IOU is over 0.5. That is when the loss is to be applied
	mask = groundTruth[:, :, 0]
	# transform the ground truth
	gx = (groundTruth[:, :, 1] - anchors[:, 0]) / anchors[:, 2]
	gy = (groundTruth[:, :, 2] - anchors[:, 1]) / anchors[:, 3]
	gw = np.log(groundTruth[:, :, 3] / anchors[:, 2])
	gh = np.log(groundTruth[:, :, 4] / anchors[:, 3])

	loss = smoothL1Debug(gx, predictions[:, :, 0], mask) +\
			smoothL1Debug(gy, predictions[:, :, 1], mask) +\
			smoothL1Debug(gw, predictions[:, :, 2], mask) +\
			smoothL1Debug(gh, predictions[:, :, 3], mask)
	return loss * alpha
#######################################################################################################


class Model():
	def __init__(self, _alpha = 1.):
		global anchors, alpha
		alpha = _alpha
		self.alpha = _alpha
		self.anchors = anchors
		inputs = keras.layers.Input(shape=(128, 128, 3), name = "first_input")
		features16, features8 = BlazeNet(inputs)
		output = SSD(features16, features8)
		self.model = tf.keras.models.Model(inputs=inputs, outputs=output)
		optimizer = tf.keras.optimizers.Adam(amsgrad=True)
		self.model.compile(loss=['binary_crossentropy', boundingBoxLoss], optimizer=optimizer)

	def train(self):
		# Taken from https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
		global trainingDataset, valDataset
		self.model.fit_generator(generator = trainingDataset,
						   steps_per_epoch = len(trainingDataset),
						   epochs = 50,
						   verbose = 1,
						   validation_data = valDataset,
						   validation_steps = len(valDataset),
						   shuffle = True,
						   use_multiprocessing = False
						)

	def eval(self, dataset):
		dataset.on_epoch_end() 
		batchSize = dataset.batchSize
		numBatches = (dataset.numSamples + batchSize - 1) // batchSize
		for batchNumber in range(numBatches):
			currentInput, (groundTruthLabels, groundTruthBox) = dataset.__getitem__(batchNumber)
			predictedLabels, predictedBoxes = self.model.predict_on_batch(currentInput)
			predictedLabels = predictedLabels.numpy()
			predictedBoxes = predictedBoxes.numpy()
			# loss = boundingBoxLossDebug(groundTruthBox, predictedBoxes)
			for i, (singlePredictedLabels, singlePredictedBoxes) in enumerate(zip(predictedLabels, predictedBoxes)):
				finalBoundingBox = self.eliminateMultiple(singlePredictedLabels, singlePredictedBoxes)
				finalBoundingBox = dataset.convertCentreSideToBounds(finalBoundingBox)
				singleGroundTruth = dataset.convertCentreSideToBounds(groundTruthBox[i, 0, 1:])
				# print(list(zip(finalBoundingBox, singleGroundTruth))[0])
				self.dumpImage(currentInput[i], singleGroundTruth, finalBoundingBox, \
					os.path.join('../data/predictions', str(batchNumber * dataset.batchSize + i) + '.jpg'))

	def evaluate(self, dataset):
		self.model.evaluate_generator(generator = dataset,
						   verbose = 1,
						   use_multiprocessing = False
						)

	def eliminateMultiple(self, labels, boxes):
		# Change this function to take into consideration multiple boxes which need to be averaged
		maxIndex = np.argmax(labels)
		return self.scaleToCentreSide(boxes[maxIndex], self.anchors[maxIndex])

	def scaleToCentreSide(self, modelOutput, anchor):
		# Model outputs need to be scaled to actually arrive at the centre and sides of the box
		assert modelOutput.shape == anchor.shape, print(modelOutput.shape, anchor.shape)
		cx = anchor[0] + anchor[2] * modelOutput[0]
		cy = anchor[1] + anchor[3] * modelOutput[1]
		w = anchor[2] * tf.math.exp(modelOutput[2])
		h = anchor[3] * tf.math.exp(modelOutput[3])
		return [cx, cy, w, h]

	def dumpImage(self, image, groundTruth, prediction, filename):
		groundTruth = [int(x + 0.5) for x in groundTruth]
		prediction = [int(x + 0.5) for x in prediction]
		print(list(zip(prediction, groundTruth)))
		image = (image + 1.0) * 127.5
		image = cv2.rectangle(image, (groundTruth[0], groundTruth[1]), (groundTruth[2], groundTruth[3]),(255, 0, 0), thickness = 1)
		image = cv2.rectangle(image, (prediction[0], prediction[1]), (prediction[2], prediction[3]),(0, 255, 0), thickness = 1)
		cv2.imwrite(filename, image)


######################################## INITILIAZATION ############################################
trainingDataset, valDataset, testDataset = None, None, None
anchors = None
alpha = None

def generateAnchors():
	# CHECK - Not sure about this stuff!
	global anchors
	anchors = np.zeros((896, 4))
	# anchors stores cx, cy, w and h in that order for every entry
	## First 2 resolutions at 16x16 - 128/16 = 8, so 6 and 10
	resolutions = [6.0, 10.0]
	count = 0
	for i in range(16):
		for j in range(16):
			for resolution in resolutions:
				anchors[count, 0] = (i + 0.5) * (128.0 / 8)
				anchors[count, 1] = (j + 0.5) * (128.0 / 8)
				anchors[count, 2:] = resolution
				count += 1

	resolutions = [16., 24., 48., 64., 80., 100.]
	for i in range(8):
		for j in range(8):
			for resolution in resolutions:
				anchors[count, 0] = (i + 0.5) * (128.0 / 8)
				anchors[count, 1] = (j + 0.5) * (128.0 / 8)
				anchors[count, 2:] = resolution
				count += 1

	assert count == 896

def init(trainBatchSize, testBatchSize):
	global trainingDataset, valDataset, testDataset, anchor
	generateAnchors()
	trainingDataset = DataGenerator('train', anchors, batchSize = trainBatchSize, shuffle = False)
	valDataset = DataGenerator('val', anchors, batchSize = testBatchSize, shuffle = False)
	testDataset = DataGenerator('test', anchors, batchSize = testBatchSize, shuffle = False)
	return

#######################################################################################################




if __name__ == "__main__":
	np.random.seed(7)
	# Some things taken from https://github.com/benuri/BlazeFace.git
	trainBatchSize, testBatchSize = 8, 16
	print("Initializing...")
	init(trainBatchSize, testBatchSize)
	print("Initialization Complete!")
	print("Preparing Model...")
	model = Model(_alpha = 0.01)
	# print(model.model.summary())
	print("Model prepared")
	# model.evaluate(trainingDataset)
	model.eval(valDataset)
	# model.evaluate(testDataset)
	# model.train()
	# model.evaluate(trainingDataset)
	# model.evaluate(valDataset)
	# model.evaluate(testDataset)
	# model.model.save_weights('../models/try1')

	# model.model.load_weights('../models/try1')
	# model.evaluate(trainingDataset)
	# model.evaluate(valDataset)
	# model.evaluate(testDataset)

