import tensorflow as tf
from tensorflow import keras
import numpy as np
from network import BlazeNet, SSD
from dataset import DataGenerator
import cv2
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

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

def weighted_categorical_crossentropy(weights):
    weights = tf.keras.backend.variable(weights)
        
    def loss(y_true, y_pred):
        # clip to prevent NaN's and Inf's
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        loss = y_true * tf.keras.backend.log(y_pred) * weights
        loss = -tf.keras.backend.sum(loss, -1)
        return loss
    
    return loss


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
        def __init__(self, _alpha = 1., numClasses = 10):
                global anchors, alpha
                alpha = _alpha
                self.numClasses = numClasses
                self.alpha = _alpha
                self.anchors = anchors
                inputs = keras.layers.Input(shape=(128, 128, 1), name = "first_input")
                features16, features8 = BlazeNet(inputs)
                output = SSD(features16, features8, numClasses)
                self.model = tf.keras.models.Model(inputs=inputs, outputs=output)
                optimizer = tf.keras.optimizers.Adam(amsgrad=True)
                weights = np.array([3.] * self.numClasses + [1./3])
                self.model.compile(loss=[weighted_categorical_crossentropy(weights), boundingBoxLoss], optimizer=optimizer)

        def train(self, numEpochs):
                # Taken from https://medium.com/@mrgarg.rajat/training-on-large-datasets-that-dont-fit-in-memory-in-keras-60a974785d71
                global trainingDataset, valDataset
                self.model.fit_generator(generator = trainingDataset,
                                                   steps_per_epoch = len(trainingDataset),
                                                   epochs = numEpochs,
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
                                classPrediction, finalBoundingBox = self.eliminateMultiple(singlePredictedLabels, singlePredictedBoxes)
                                print(classPrediction, groundTruthLabels[i, :, :self.numClasses].argmax() % self.numClasses)
                                finalBoundingBox = dataset.convertCentreSideToBounds(finalBoundingBox)
                                singleGroundTruth = dataset.convertCentreSideToBounds(groundTruthBox[i, 0, 1:])
                                # print(list(zip(finalBoundingBox, singleGroundTruth))[0])
                                self.dumpImage(currentInput[i], singleGroundTruth, finalBoundingBox, classPrediction, \
                                        os.path.join('../predictions', str(batchNumber * dataset.batchSize + i) + '.jpg'))

        def evaluate(self, dataset):
                self.model.evaluate_generator(generator = dataset,
                                                   verbose = 1,
                                                   use_multiprocessing = False
                                                )

        def eliminateMultiple(self, labels, boxes):
                # Change this function to take into consideration multiple boxes which need to be averaged
                labels = labels[:, :self.numClasses]
                argmax = np.argmax(labels)
                maxIndex = argmax // (self.numClasses)
                classPrediction = argmax % (self.numClasses)
                return classPrediction, self.scaleToCentreSide(boxes[maxIndex], self.anchors[maxIndex])

        def scaleToCentreSide(self, modelOutput, anchor):
                # Model outputs need to be scaled to actually arrive at the centre and sides of the box
                assert modelOutput.shape == anchor.shape, print(modelOutput.shape, anchor.shape)
                cx = anchor[0] + anchor[2] * modelOutput[0]
                cy = anchor[1] + anchor[3] * modelOutput[1]
                w = anchor[2] * tf.math.exp(modelOutput[2])
                h = anchor[3] * tf.math.exp(modelOutput[3])
                return [cx, cy, w, h]

        def dumpImage(self, greyscaleImage, groundTruth, prediction, classPrediction, filename):
                groundTruth = [int(x + 0.5) for x in groundTruth]
                prediction = [int(x + 0.5) for x in prediction]
                # print(list(zip(prediction, groundTruth)))
                greyscaleImage = (greyscaleImage + 1.0) * 127.5
                image = np.zeros((128, 128, 3))
                image[:, :, 0:1] = greyscaleImage
                image[:, :, 1:2] = greyscaleImage
                image[:, :, 2:3] = greyscaleImage
                image = cv2.rectangle(image, (groundTruth[0], groundTruth[1]), (groundTruth[2], groundTruth[3]),(255, 0, 0), thickness = 1)
                image = cv2.rectangle(image, (prediction[0], prediction[1]), (prediction[2], prediction[3]),(0, 255, 0), thickness = 1)
                cv2.putText(image, str(classPrediction),(0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)
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
        with tf.device("/gpu:0"):
                np.random.seed(7)
                # Some things taken from https://github.com/benuri/BlazeFace.git
                trainBatchSize, testBatchSize = 8, 16
                print("Initializing...")
                init(trainBatchSize, testBatchSize)
                print("Initialization Complete!")
                print("Preparing Model...")
                model = Model(_alpha = 0.)
                # model.model.summary()
                print("Model prepared")
                model.evaluate(trainingDataset)
                model.evaluate(valDataset)
                model.evaluate(testDataset)
                model.train(5)
                model.evaluate(trainingDataset)
                model.evaluate(valDataset)
                model.evaluate(testDataset)
                model.eval(testDataset)
                model.model.save_weights('../models/try0')

       
                print("Saved model")
                model2 = Model(_alpha = 0.01)
                model2.evaluate(testDataset)
                model2.train(5)
                model2.evaluate(testDataset)
                model2.eval(testDataset)
                model2.model.save_weights('../models/try1_5')
                model2.train(5)
                model2.evaluate(testDataset)
                model2.eval(testDataset)
                model2.model.save_weights('../models/try1_10')
                # model.model.load_weights('../models/try1')
                # model.evaluate(trainingDataset)
                # model.evaluate(valDataset)
                # model.evaluate(testDataset)
