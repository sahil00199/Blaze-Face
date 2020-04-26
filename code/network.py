import tensorflow as tf
from tensorflow import keras
import numpy as np

############################################## MODELS ##################################################
def SingleBlazeBlock(input, outputChannels, stride = 1, prefix = "SingleBB"):
	x = keras.layers.DepthwiseConv2D(5, strides=stride, padding='same', data_format="channels_last", name = prefix + "_DW")(input)
	x = keras.layers.Conv2D(outputChannels, 1, strides=(1, 1), padding='same', data_format='channels_last', name = prefix + "_1x1")(x)
	x = keras.layers.BatchNormalization(name = prefix + '_batchNorm')(x)

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
	x = keras.layers.BatchNormalization(name = prefix + '_batchNorm1')(x)
	x = keras.layers.Activation("relu", name = prefix + "_activation1")(x)
	x = keras.layers.DepthwiseConv2D(5, strides=1, padding='same', data_format="channels_last", name = prefix + "_DW2")(x)
	x = keras.layers.Conv2D(outputChannels, 1, strides=(1, 1), padding='same', data_format='channels_last', name = prefix + "_expand")(x)
	x = keras.layers.BatchNormalization(name = prefix + '_batchNorm2')(x)

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
	x1 = keras.layers.BatchNormalization()(x1)
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
	boundingBox = tf.keras.backend.clip(boundingBox, -1, 1)
	classLabels = tf.keras.layers.Concatenate(axis=1, name='concat_classLabels')([classLabels16, classLabels8])
	return classLabels, boundingBox




	
