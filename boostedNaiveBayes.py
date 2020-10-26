

import math
import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random
from sklearn import tree



def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    for i in range(Nclasses):
        idx = np.where(labels==classes[i])[0]
        weightsForClass = np.array(W[idx,:])
        prior[i] = np.sum(weightsForClass)


    return prior



def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))
    counter = 0 # used to keep track of the current class

    for _class in classes:
         idx = np.where(labels==_class)[0]
         xlc = np.array(X[idx,:]) # Get the x for the class labels. Vectors are rows.
         weightsForClass = np.array(W[idx,:])
         weightedVectors= xlc * weightsForClass
         mu[counter] =  np.sum(weightedVectors, axis = 0) / np.sum(weightsForClass)
         matrixOfSquares = ((xlc - mu[counter])**2) * weightsForClass
         diagonalVector = np.sum(matrixOfSquares, axis = 0) / np.sum(weightsForClass)
         np.fill_diagonal(sigma[counter], diagonalVector)
         counter += 1

    return mu, sigma

def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    for i in range(Nclasses):
        logarithmOfDetCov = -0.5 * (math.log(np.linalg.det(sigma[i])))
        logarithmOfprior = math.log(prior[i])
        constantTermsForClass = logarithmOfprior + logarithmOfDetCov
        feautureMinusAverageMatrix = X - mu[i]
        inverseDiagonal = np.linalg.inv(sigma[i])
        for j in range(Npts):
          logProb[i][j] = -0.5 * np.matmul(np.matmul(feautureMinusAverageMatrix[j], inverseDiagonal), np.transpose(feautureMinusAverageMatrix[j])) + constantTermsForClass
    h = np.argmax(logProb,axis=0)
    return h



class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)



def classifyVector(votes, labels):
    res = np.zeros((len(labels),1))
    for i in range(len(labels)):
        res[i] = 0 if votes[i] == labels[i] else 1
    return res

    def classifyVectorUnchanged(votes, labels):
        res = np.zeros((len(labels),1))
        for i in range(len(labels)):
            res[i] = 1 if votes[i] == labels[i] else 0
        return res

def computeNewWeights(votes, labels, oldWeights, currentAlpha):
    firstWeights = np.zeros((len(labels),1))
    for i in range(len(labels)):
        firstWeights[i] = oldWeights[i] * (math.exp(-currentAlpha) if votes[i] == labels[i] else math.exp(currentAlpha))
    return firstWeights / float(np.sum(firstWeights))



def trainBoost(base_classifier, X, labels, T=10):
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))
        # do classification for each point
        vote = classifiers[-1].classify(X)
        errorSum = np.sum((classifyVector(vote, labels) * wCur ))
        currentAlpha = 0.5 * (math.log(1 - errorSum) - math.log(errorSum + 1e-10))
        alphas.append(currentAlpha)
        wCur = computeNewWeights(vote, labels, wCur, currentAlpha)
    return classifiers, alphas


def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)
    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))
        matrix = []
        for classifier in classifiers:
            matrix.append(classifier.classify(X))
        for i in range(len(X)):
            for j in range(Nclasses):
               for k in range(len(alphas)):
                   votes[i][j] += alphas[k] * (1 if (matrix[k][i] == j) else 0)
        return np.argmax(votes,axis=1)


class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)
