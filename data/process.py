import pickle
import numpy as np
import cv2 

annotations = {}
images = {}

count = 0
for inputFile in ['mnist_test.csv', 'mnist_train.csv']:
	file = open(inputFile, 'r')
	lines = file.readlines()
	file.close()
	for line in lines:
		line = line.strip().split(',')
		if len(line) == 0:
			continue
		line = [int(x.strip()) for x in line[1:]]
		image = np.zeros((128, 128))
		number = np.array(line).reshape(28, 28).astype('float32')
		finalShape = np.random.randint(16, 80)
		number = cv2.resize(number, (finalShape, finalShape))
		x = np.random.randint(0, 128 - finalShape)
		y = np.random.randint(0, 128 - finalShape)
		image[x:x + finalShape, y:y + finalShape] = number
		image = image.reshape(128, 128, 1)
		images[count] = image
		annotations[count] = [x, y, finalShape, finalShape]
		count += 1


