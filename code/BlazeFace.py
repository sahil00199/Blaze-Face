import tensorflow as tf
from tensorflow import keras
import numpy as np

############################################### DATA GENERATION #####################################################
class DataGenerator(keras.utils.Sequence):
	# taken from: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
	def __init__(self, numSamples, batchSize=32, shuffle=True):
		'Initialization'
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
		global anchors
		return np.asarray([1 if self.iou(self.convertCentreSideToBounds(anchor), boundingBox) > 0.5 else 0 for anchor in anchors])

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




##########################################  LOSS ################################################
def smoothL1(groundTruth, predictions, globalMask):
	# compute smooth L1
	diff = predictions - groundTruth
	mask = tf.cast(tf.math.abs(diff) < 1, dtype=tf.float32)
	loss = ((diff ** 2) * 0.5) * mask + (tf.math.abs(diff) - 0.5) * (1.0 - mask)
	loss = tf.keras.backend.mean(loss * globalMask)
	return loss
	
def boundingBoxLoss(groundTruth, predictions):
	global anchors
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
	return loss

def smoothL1Debug(groundTruth, predictions, globalMask):
	# compute smooth L1
	diff = predictions - groundTruth
	mask = np.cast[np.float32](np.abs(diff) < 1)
	loss = ((diff ** 2) * 0.5) * mask + (np.abs(diff) - 0.5) * (1.0 - mask)
	loss = np.mean(loss * globalMask)
	return loss
	
def boundingBoxLossDebug(groundTruth, predictions):
	global anchors
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
	print(tf.keras.backend.get_value(gh[0, 682]), tf.keras.backend.get_value(predictions[0, 682, 3]))
	print(tf.keras.backend.get_value(gw[0, 682]), tf.keras.backend.get_value(predictions[0, 682, 2]))
	print(tf.keras.backend.get_value(gy[0, 682]), tf.keras.backend.get_value(predictions[0, 682, 1]))
	print(tf.keras.backend.get_value(gx[0, 682]), tf.keras.backend.get_value(predictions[0, 682, 0]))
	return loss

#######################################################################################################





############################################## MODELS ##################################################
def SingleBlazeBlock(input, outputChannels, stride = 1, prefix = "SingleBB"):
	x = keras.layers.DepthwiseConv2D(5, strides=stride, padding='same', data_format="channels_last", name = prefix + "_DW")(input)
	x = keras.layers.Conv2D(outputChannels, 1, strides=(1, 1), padding='same', data_format='channels_last', name = prefix + "_1x1")(x)
	# x = keras.layers.BatchNormalization(name = prefix + '_batchNorm')(x)

	if stride == 2:
		original = tf.keras.layers.MaxPooling2D(name = prefix + "_MaxPool")(input)
		if input.shape[-1] != outputChannels:
			original = tf.keras.layers.Concatenate(axis=-1, name = prefix + "_concat")([original, tf.zeros_like(original)]) # CHECK - not sure if we should pad with 0s or concatenate input again
	else:
		original = input
	
	output = tf.keras.layers.Add(name = prefix + '_add')([x, original])
	output = tf.keras.layers.Activation('relu', name = prefix + "_activation")(output)
	return output

def DoubleBlazeBlock(input, outputChannels, intermediateChannels, stride = 1, prefix = "DoubleBB"):
	x = keras.layers.DepthwiseConv2D(5, strides=stride, padding='same', data_format="channels_last", name = prefix + "_DW1")(input)
	x = keras.layers.Conv2D(intermediateChannels, 1, strides=(1, 1), padding='same', data_format='channels_last', name = prefix + "_project")(x)
	# x = keras.layers.BatchNormalization(name = prefix + '_batchNorm1')(x)
	x = keras.layers.Activation("relu", name = prefix + "_activation1")(x)
	x = keras.layers.DepthwiseConv2D(5, strides=1, padding='same', data_format="channels_last", name = prefix + "_DW2")(x)
	x = keras.layers.Conv2D(outputChannels, 1, strides=(1, 1), padding='same', data_format='channels_last', name = prefix + "_expand")(x)
	# x = keras.layers.BatchNormalization(name = prefix + '_batchNorm2')(x)

	if stride == 2:
		original = tf.keras.layers.MaxPooling2D(name = prefix + "_MaxPool")(input)
		if input.shape[-1] != outputChannels:
			original = tf.keras.layers.Concatenate(axis=-1, name = prefix + "_concat")([original, tf.zeros_like(original)]) # CHECK - not sure if we should pad with 0s or concatenate input again
	else:
		original = input
	
	output = tf.keras.layers.Add(name = prefix + '_add')([x, original])
	output = tf.keras.layers.Activation('relu', name = prefix + "_activation2")(output)
	return output


def BlazeNet(x):
	x1 = keras.layers.Conv2D(24, 5, strides=(2, 2), padding='same', data_format='channels_last', name = "initial_conv")(x)
	# x1 = keras.layers.BatchNormalization()(x1)
	x1 = keras.layers.Activation("relu")(x1)

	x2 = SingleBlazeBlock(x1, outputChannels = 24, prefix = "Single1")
	x2 = SingleBlazeBlock(x2, outputChannels = 24, prefix = "Single2")
	x3 = SingleBlazeBlock(x2, outputChannels = 48, stride=2, prefix = "Single3")

	x3 = SingleBlazeBlock(x3, outputChannels = 48, prefix = "Single4")
	x3 = SingleBlazeBlock(x3, outputChannels = 48, prefix = "Single5")
	x4 = DoubleBlazeBlock(x3, outputChannels = 96, intermediateChannels = 24, stride=2, prefix = "Double1")

	x4 = DoubleBlazeBlock(x4, outputChannels = 96, intermediateChannels = 24, prefix = "Double2")
	x4 = DoubleBlazeBlock(x4, outputChannels = 96, intermediateChannels = 24, prefix = "Double3")
	x5 = DoubleBlazeBlock(x4, outputChannels = 96, intermediateChannels = 24, stride = 2, prefix = "Double4")
	
	x5 = DoubleBlazeBlock(x5, outputChannels = 96, intermediateChannels = 24, prefix = "Double5")
	x5 = DoubleBlazeBlock(x5, outputChannels = 96, intermediateChannels = 24, prefix = "Double6")

	return [x4, x5]

def SSD(features16, features8):
	# For each anchor, 5 outputs are needed
	preds16 = keras.layers.Conv2D(2 * 5, 3, strides=1, padding='same', data_format='channels_last', name = "16x16_conv")(features16)
	preds16 = keras.layers.Reshape((16 * 16 * 2, 5), name = "16x16_reshape")(preds16)
	classLabels16 = keras.layers.Activation('sigmoid', name='16x16_activation')(preds16[:, :, :1])
	boundingBox16 = preds16[:, :, 1:]

	preds8 = keras.layers.Conv2D(6 * 5, 3, strides=1, padding='same', data_format='channels_last', name = "8x8_conv")(features8)
	preds8 = keras.layers.Reshape((8 * 8 * 6, 5), name = "8x8_reshape")(preds8)
	classLabels8 = keras.layers.Activation('sigmoid', name = "8x8_activation")(preds8[:, :, :1])
	boundingBox8 = preds8[:, :, 1:]

	boundingBox = tf.keras.layers.Concatenate(axis=1, name='concat_bounding_box')([boundingBox16, boundingBox8])
	classLabels = tf.keras.layers.Concatenate(axis=1, name='concat_classLabels')([classLabels16, classLabels8])
	return classLabels, boundingBox

class Model():
	def __init__(self):
		global anchors
		self.anchors = anchors
		inputs = keras.layers.Input(shape=(128, 128, 3), name = "first_input")
		features16, features8 = BlazeNet(inputs)
		output = SSD(features16, features8)
		self.model = tf.keras.models.Model(inputs=inputs, outputs=output)
		optimizer = tf.keras.optimizers.Adam(amsgrad=True)
		self.model.compile(loss=['binary_crossentropy', boundingBoxLoss], optimizer=optimizer)

	def train(self):
		# Taken from https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
		global trainingDataset, testDataset
		self.model.fit_generator(generator = trainingDataset,
						   steps_per_epoch = len(trainingDataset),
						   epochs = 50,
						   verbose = 1,
						   validation_data = testDataset,
						   validation_steps = len(testDataset),
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
			print(boundingBoxLossDebug(groundTruthBox, predictedBoxes))
			for i, (singlePredictedLabels, singlePredictedBoxes) in enumerate(zip(predictedLabels, predictedBoxes)):
				finalBoundingBox = self.eliminateMultiple(singlePredictedLabels, singlePredictedBoxes)
				finalBoundingBox = dataset.convertCentreSideToBounds(finalBoundingBox)
				singleGroundTruth = dataset.convertCentreSideToBounds(groundTruthBox[i, 0, 1:])
				print(list(zip(finalBoundingBox, singleGroundTruth))[0])

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



#######################################################################################################




######################################## INITILIAZATION ############################################
anchors = None
trainingDataset, testDataset = None, None

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
	global trainingDataset, testDataset
	generateAnchors()
	trainingDataset = DataGenerator(8, batchSize = trainBatchSize, shuffle = False)
	testDataset = DataGenerator(32, batchSize = testBatchSize, shuffle = False)
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
	model = Model()
	# print(model.model.summary())
	print("Model prepared")
	# print("Training...")
	# model.train()
	# print("Training Done!")
	print("Running inference...")
	for i in range(1):
		print('*' * 75 + str(i) + '*' * 75)
		model.train()
		model.eval(trainingDataset)
	print("Inference done")
