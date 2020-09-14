# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

import numpy as np

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set
"""

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):

    # ARRAY OF WEIGHTS FOR EACH IMAGE (1 to 1 with size of a picture)
    array_of_weights = np.zeros(len(train_set[0]))

    # INITIALIZE BIAS TO 0
    bias = 0

    # CALCULATE WEIGHTS FROM TRAIN SET
    for iteration_number in range (1, max_iter + 1):
        for image, type_of_pic in zip(train_set, train_labels):
            if(np.dot(image, array_of_weights[0:]) + bias) <= 0:
                image_prediction = 0
            else:
                image_prediction = 1
            constant = learning_rate * (type_of_pic - image_prediction)
            array_of_weights[0:] += constant * image
            bias += constant

    # RETURN WEIGHT AND BIAS
    return array_of_weights, bias

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set

    weight_array, bias = trainPerceptron(train_set, train_labels, learning_rate, max_iter)

    final_predictions = []

    for every_image in dev_set:
        if (np.dot(every_image, weight_array[0:]) + bias) <= 0:
            image_prediction = 0
        else:
            image_prediction = 1
        final_predictions.append(image_prediction)

    return final_predictions

def sigmoid(x):
    # TODO: Write your code here
    # return output of sigmoid function given input x
    return 1 / (1 + np.exp(-x))

def trainLR(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    # ARRAY OF WEIGHTS FOR EACH IMAGE (1 to 1 with size of a picture)

    #print("SIGMOID: ", sigmoid(10))
    array_of_weights = np.zeros(len(train_set[0]))

    # INITIALIZE BIAS TO 0
    bias = 0

    # CALCULATE WEIGHTS FROM TRAIN SET
    for iteration_number in range (1, max_iter + 1):
        for image, type_of_pic in zip(train_set, train_labels):
            if(np.dot(image, array_of_weights[0:]) + bias) <= 0:
                image_prediction = 0
            else:
                image_prediction = 1
            constant = learning_rate * (type_of_pic - image_prediction)
            array_of_weights[0:] += constant * image
            bias += constant

    # RETURN WEIGHT AND BIAS

    return array_of_weights, bias

def classifyLR(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train LR model and return predicted labels of development set
    weight_array, bias = trainPerceptron(train_set, train_labels, learning_rate, max_iter)

    final_predictions = []

    for every_image in dev_set:
        if (np.dot(every_image, weight_array[0:]) + bias) <= 0:
            image_prediction = 0
        else:
            image_prediction = 1
        final_predictions.append(image_prediction)

    return final_predictions

def classifyEC(train_set, train_labels, dev_set, k):
    # Write your code here if you would like to attempt the extra credit
    weight_array, bias = trainLR(train_set, train_labels, 1e-2, 10)

    final_predictions = []

    for every_image in dev_set:
        if (np.dot(every_image, weight_array[0:]) + bias) <= 0:
            image_prediction = 0
        else:
            image_prediction = 1
        final_predictions.append(image_prediction)
    return final_predictions
