import pickle
import numpy as np
import cv2 

annotations = {}
images = {}

count = 0
for split in ['train', 'test', 'val']:
	inputFile = 'mnist_' + split + '.csv'
	annotations[split] = {}
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
		image = image.reshape(128, 128, 1).astype('uint8')
		images[count] = image
		annotations[split][count] = [y, x, finalShape, finalShape]
		count += 1
		if count == 25000: break
		if count == 30000: break
		if count == 35000: break

pickle.dump(images, open("processed/images.pkl", 'wb'))
pickle.dump(annotations, open("processed/annotations.pkl", 'wb'))


