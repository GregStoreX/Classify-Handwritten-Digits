# Classify-Handwritten-Digits

Classify handwritten digits using the famous MNIST data

## Overview

The goal of this project is to take an image of a handwritten single digit, and determine what that digit is. From the provided dataset, each row in training dataset means the label and 28x28 pixels data of an image. The MNIST database is a large database of handwritten digits that is commonly used for training various image processing systems, and through the home page of this dataset, we found that it provides many kinds of solution implemented by different machine learning algorithms without random forest. Random Forest uses multiple classifiers for bagging, boosting and random subspaces, and there is a growing interest to use this kind of multiple classifier system in pattern recognition field. Based on previously works, we want to implement random forest by coding so as to have a deeper understand about how a machine learning algorithm works in real life problem. 
Additionally, we use third-part toolbox, like WEKA, to train and evaluate the same dataset with random forest and other machine learning algorithms, so that we can compare the performances between our implement and existing solutions. The evaluation metric for this project is the categorization accuracy, or the proportion of test images that are correctly classified. Besides, some performance results will be presented to evaluate this project.

## Dataset Description

The MNIST (Modified National Institute of Standards and Technology) database gives a lot of images of handwritten digits. It is constructed based on NIST (National Institute of Standards and Technology) which gives data set of over 800,000 images of handwritten digits from 3,600 writers. The MNIST data have been size-normalized and centered in a fixed size image, so we can use this data and apply our learning algorithms to real-world problem without additional effort on pre-process procedure. We use the dataset which is organized for Keggle competition.
The data contains gray-scale images of hand-written digits (0~9). The each image consists of total 784 pixels (28 * 28) and pixel-value (0~255) which determines the degree of darkness or lightness. The image data is pre-processed and combined to one CSV type. The column of CSV data means pixels of an image data and the row means each image data. Training data set contains label, but test data set doesn’t contain the label.


Features of the dataset are as below.
1. Real hand-printed data prepared by individuals (approximately 500 writers)
2. Amount of data: 40,000 examples (71.4MB)
3. Amount of features: 784 (pixel)
4. CSV data type

## Methods

### Feature Selection

Due to the observation that the input dataset has enormous numbers of dimensions, 784 features representing the grey value in each pixel, it is computationally expensive to directly put the raw data into the classifiers. Hence we present two feature selection methods, Principal Component Analysis and Convolution.

### Random Forest

In this project, we tried to implement our own Random Forest without any machine learning toolbox provided by MATLAB, which is a famous a high-level language and interactive environment for numerical computation, visualization, and programming. In this part, we will discuss how to implement a Random Forest by MATLAB_R2015a to predict label values in MNIST dataset.

### Logistic Regression

we apply the logistic regression on the transformed dataset of 32-dim, 64-dim and 128-dim and show the f1 scores and time consumption of each dimensionality choice. we observe that the time consumption for model training increased dramatically from 64-dim to 128 dim while the increased dimensionality doesn’t pay back greatly, only a slight increase in f1 measure and accuracy. Hence we use the first 64 eigen-vectors to form a new feature space.

### Simulation on WEKA

Weka is a famous machine learning tool written in Java. This is open source software under the GPL. It helps to solve data science problem like classification, cluster and associate by giving a lot of useful machine learning algorithms library. Representative algorithms in the data science field are like Decision Tree, MLP, Naïve Bayes and Random Forest. Each algorithm has distinct characteristics because they are based on different theory as Tree, Neural Network, Bayes Theory (Probability) and Ensemble. In this paper, we show the difference of these algorithms in the handwritten digits recognition problem. For comparing the performance of these algorithms, we use the algorithms implemented in Weka. The basic setting of parameters of the algorithms is as follows.

• Decision Tree (J48) -> Confidence Factor: 0.25, Minimum number of objects: 2 and Number of Folds: 3 
• MLP -> Hidden Layer: 1, Learning rate: 0.3, Momentum: 0.2, Training time (The number of epochs): 100 and Validation Threshold: 20
• Naïve Bayes-> None
• Random Forest-> Maximum depth: 10, Number of Random Features: 10 and Number of Trees: 100 



















