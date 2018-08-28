# coding:utf-8
import math
from numpy import digitize, linspace
import random
import sklearn.metrics as skm
import pandas as pd

def prob(e, space):
    p = float(space.count(e))/len(space)
    return p

def weigthed_mutual_information(x, y, weights):
    x = list(x)
    y = list(y)
    if len(weights) != len(x[0]):
        raise ValueError("There must be as many features as there are weights sets")
    xT = [list(column) for column in zip(*x)]
    uY = set(y)
    mi = [0] * len(xT)
    for index, column in enumerate(xT):
        uX = set(column)
        joint = list(zip(y,column))
        for vY in uY:
            for i, vX in enumerate(column):
                w = weights[index][vX] 
                p = prob((vY,vX),joint)
                prob_ratio = float(p)/(prob(vX,column)*prob(vY,y))
                if p > 0:
                    mi[index] +=  w * p * math.log(prob_ratio)
    return mi

# Used to encapsulate all data and metadata for message passing between layers in mime
class DataWrapper(object):

    def __init__(self, data, target, categorical):
        self.data = data
        self.categorical = categorical
        self.target = target

# Before explaining our instances we may need to do
# some pre-calculations for some reason, this is
# the role of the preprocessor
class MimePreprocessor(object):

    # You may use kwargs to set some parameters for the preprocessor
    def __init__(self, data, **kwargs):
        self.data = data
        self.args = kwargs
        self.computed = kwargs

    def preprocess(self, data=None):
        if data==None:
            workingData = self.data
        else:
            workingData = data
        # For MIME we need to precompute the optimal number of bins for each non-categorical feature
        # We do this using Sturge's Formula and the Sample Size
        # Then we change our data to the binned version
        self.computed["optimalBins"] = int(math.log(workingData.data.shape[0], 2) + 1)
        self.computed["sampleSize"] = int(self.args["sampleFrac"] * workingData.data.shape[0])
        self.computed["bins"] = []
        dataCopy = workingData.data.copy()
        tmp = workingData.data.T.values
        for i in range(workingData.data.shape[1]):
            if i in workingData.categorical:
                continue
            #Digitizes the values on the given columns
            bins = linspace(min(tmp[i]), max(tmp[i]), self.computed["optimalBins"])
            self.computed["bins"].append(bins)
            binned = digitize(tmp[i], bins)
            dataCopy[dataCopy.columns[i]] = binned
        newData = DataWrapper(dataCopy, workingData.target, workingData.categorical)
        return newData

class Sampler(object):

    def __init__(self):
        random.seed()

    def sample_neighborhood(self, instance, data, size):
        neighborhood = []
        means = data.data.mean()
        stdevs = data.data.std()
        for i in range(size):
            sample = [means[feature] + random.normalvariate(0,1) * stdevs[feature] for feature in range(len(data.data.columns))]
            neighborhood.append(sample)
        return neighborhood

# At last, the explainer takes an instance and pre-computed data
# and generates an explanation for it
class MimeExplainer(object):

    def __init__(self, sampler=Sampler(), **kwargs):
       self.args = kwargs
       self.sampler = sampler

    # You can use kwargs to pass parameters precomputed by the preprocessor
    # MimeExplainer expects a "sampleSize" parameter
    def explain(self, instance, data, predictor, preprocessor):
        # Preprocess data
        processedData = preprocessor.preprocess()
        # Generate neighborhood samples
        neighborhood = pd.DataFrame(self.sampler.sample_neighborhood(instance, data, preprocessor.computed['sampleSize']));
        # Label the samples
        neighborhoodLabels = predictor(neighborhood)
        # Weight the samples
        bins_weights = self.weight_bins(instance, preprocessor.computed["bins"])
        neighborhoodData = DataWrapper(neighborhood, neighborhoodLabels, data.categorical)
        neighborhoodData = preprocessor.preprocess(neighborhoodData)
        # For each feature, calculates its mutual information with the labels
        return weigthed_mutual_information(neighborhoodData.data.values, neighborhoodLabels, bins_weights), predictor([instance])

    def weight_bins(self, instance, bins):
        w = []
        width = 10
        for f in range(len(instance)):
            bins_w = []
            bins_width = bins[0][1]-bins[0][0]
            for b in range(len(bins[0])):
                #Falta, gerar a última bin
                bin_midpoint = bins[f][b]-(bins_width)/2
                dsqr = (instance[f] - bin_midpoint)**2
                bins_w.append(math.exp(-dsqr/width))
            w.append(bins_w)
        return w

# MIME is a Mutual Information Model-Agnostic Explanator for machine learning
# black boxes, it uses a similar aproach to that of LIME, probing the decision
# with perturbed versions of the instance being explained
class Mime(object):
    # Mime works with pandas DataFrames
    # Parameters:
    # dataframe    : your data
    # categorical  : a list with the index of the categorical columns
    # preprocessor : the preprocessor for your explanations (default: MimePreprocessor)
    # explainer    : the explainer that will generate your explanations (default: MimeExplainer)
    # parameters   : the parameters (if any) you want to pass to your preprocessor or explainer, should be a dict
    def __init__(self, dataframe, target, categorical=[],
                 preprocessor=MimePreprocessor, explainer=MimeExplainer,
                 preprocessorParameters={"sampleFrac": 0.1}, explainerParameters={}):
        self.data = DataWrapper(dataframe, target, categorical)
        self.preprocessor = preprocessor(self.data, **preprocessorParameters)
        self.explainer = explainer(**explainerParameters)

    # Parameters:
    # instance  : the instance being explained
    # predictor : a function that takes instance and returns a black-box prediction
    def explain(self, instance, predictor):
        return self.explainer.explain(instance, self.data, predictor, self.preprocessor)

