import numpy as np
import cv2
import os
import pickle

def getAnnotations():
	allAnnotations = {}
	for split in ['train', 'val']:
		filename = 'wider_face_split/wider_face_' + split + "_bbx_gt.txt"
		allAnnotations[split] = {}
		file = open(filename, 'r')
		lines = file.readlines()
		file.close()

		index, length = 0, len(lines)
		while index < length:
			while lines[index].strip().__len__() == 0: index += 1
			imageName = lines[index].strip().split('/')[-1]
			index += 1
			numBoxes = int(lines[index].strip())
			if numBoxes == 0:
				index += 2
				continue
			index += 1
			allAnnotations[split][imageName] = [tuple([int(x) for x in line.strip().split()[:4]]) for line in lines[index:index + numBoxes]]
			index += numBoxes
	return allAnnotations

def filterAnnotations(allAnnotations):
	# Split train set into train and val (90/10 split)
	annotations = {}
	annotations['test'] = allAnnotations['val']
	allTrainKeys = list(allAnnotations['train'].keys())
	indices = np.random.permutation(len(allTrainKeys))
	trainIndices = indices[:int(0.9 * len(indices))]
	valIndices = indices[int(0.9 * len(indices)):]
	annotations['train'] = {allTrainKeys[i]: allAnnotations['train'][allTrainKeys[i]] for i in trainIndices}
	annotations['val'] = {allTrainKeys[i]: allAnnotations['train'][allTrainKeys[i]] for i in valIndices}

	return annotations

def dumpImages(annotations):
	allImages = {}
	finalAnnotations = {}
	for split in ['train', 'test', 'val']:
		finalAnnotations[split] = {}
		for imageName, singleAnnotation in annotations[split].items():
			if len(singleAnnotation) != 1: continue
			rawImage = cv2.imread(os.path.join('images', imageName))
			h, w, _ = rawImage.shape
			finalImage = cv2.resize(rawImage, (128, 128))
			cv2.imwrite(os.path.join('processed/images', imageName), finalImage)
			modifiedSingleAnnotation = [[float(x1 * 128./w), float(y1 * 128./h), float(x2 * 128./w), float(y2 * 128./h)] for x1, y1, x2, y2 in singleAnnotation]
			finalAnnotations[split][imageName] = modifiedSingleAnnotation
			allImages[imageName] = finalImage
	pickle.dump(allImages, open(os.path.join('processed', 'images.pkl'), 'wb'))
	return finalAnnotations

if __name__ == "__main__":
	np.random.seed(7)
	allAnnotations = getAnnotations()
	filteredAnnotations = filterAnnotations(allAnnotations)
	finalAnnotations = dumpImages(filteredAnnotations)
	pickle.dump(finalAnnotations, open('processed/annotations.pkl', 'wb'))




