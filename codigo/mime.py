# coding:utf-8
import math
from numpy import digitize, linspace
import random
import sklearn.metrics as skm
import pandas as pd

def prob(e, space):
    return space.count(e)/len(space)

def weigthed_mutual_information(x, y, weights):
    print(weights)
    print(x)
    print(y)
    if len(weights) != len(x):
        raise ValueError("Weights and X must have the same length")
    uX = set(x.flatten())
    uY = set(y)
    joint = list(zip(y,x))
    mi = 0
    for vY in uY:
        for vX in uX:
            w = 1 # TODO, calcular o peso para cada instância
            p = prob((vY,vX),joint)
            mi +=  w * p * math.log(p/(prob(vX,x)*prob(vY,y)))
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
        # For MIME we need toprecompute the optimal number of bins for each non-categorical feature
        # We do this using Sturge's Formula and the Sample Size
        # Then we change our data to the binned version
        self.computed["optimalBins"] = int(math.log(workingData.data.shape[0], 2) + 1)
        self.computed["sampleSize"] = int(0.1 * workingData.data.shape[0]) # 10% of the data size
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
        print(neighborhood)
        print(neighborhoodLabels)
        neighborhoodData = DataWrapper(neighborhood, neighborhoodLabels, data.categorical)
        neighborhoodData = preprocessor.preprocess(neighborhoodData)
        print(neighborhoodData.data)
        # Weight the samples

        weights = self.weight_samples(instance, neighborhoodData.data.values)
        print(weights)
        #transposedData = [list(i) for i in zip(*neighborhood)]
        # For each feature, calculates its mutual information with the labels
        return weigthed_mutual_information(neighborhoodData.data.values, neighborhoodLabels, weights), predictor([instance])

    def weight_samples(self, instance, neighborhood):
        w = []
        width = 10
        for n in neighborhood:
            dsqr = sum([(iF-nF)**2 for iF,nF in zip(instance,n)])
            w.append(math.exp(-dsqr/width))
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
                 preprocessorParameters={}, explainerParameters={}):
        self.data = DataWrapper(dataframe, target, categorical)
        self.preprocessor = preprocessor(self.data, **preprocessorParameters)
        self.explainer = explainer(**explainerParameters)

    # Parameters:
    # instance  : the instance being explained
    # predictor : a function that takes instance and returns a black-box prediction
    def explain(self, instance, predictor):
        return self.explainer.explain(instance, self.data, predictor, self.preprocessor)

