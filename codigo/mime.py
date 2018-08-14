# coding:utf-8
import math
from numpy import digitize, linspace
import random
import sklearn.metrics as skm
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

    def preprocess(self):
        # For MIME we need to precompute the optimal number of bins for each non-categorical feature
        # We do this using Sturge's Formula and the Sample Size
        # Then we change our data to the binned version
        self.computed["optimalBins"] = int(math.log(self.data.data.shape[0], 2) + 1)
        self.computed["sampleSize"] = int(0.1 * self.data.data.shape[0]) # 10% of the data size
        self.computed["bins"] = []
        dataCopy = self.data.data.copy()
        tmp = self.data.data.T.values
        for i in range(self.data.data.shape[1]):
            if i in self.data.categorical:
                continue
            #Digitizes the values on the given columns
            bins = linspace(min(tmp[i]), max(tmp[i]), self.computed["optimalBins"])
            self.computed["bins"].append(bins)
            binned = digitize(tmp[i], bins)
            dataCopy[dataCopy.columns[i]] = binned
        newData = DataWrapper(dataCopy, self.data.target, self.data.categorical)
        return newData

class Sampler(object):

    def __init__(self):
        random.seed()

    def sampleNeighborhood(self, instance, data, size):
        neighborhood = []
        means = data.data.mean()
        stdevs = data.data.std()
        for i in range(size):
            sample = [instance[feature] + random.normalvariate(0,1) * stdevs[feature] for feature in range(len(data.data.columns))]
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
       processedData = preprocessor.preprocess()

       neighborhood = self.sampler.sampleNeighborhood(instance, data, preprocessor.computed['sampleSize']);
       neighborhoodLabels = predictor(neighborhood)

       transposedData = [[processedData.data[c][l] for l in range(len(neighborhood))] for c in range(len(neighborhood[0]))]
       # For each feature, calculates its mutual information with the labels
       return [skm.mutual_info_score(transposedData[i], neighborhoodLabels) for i in range(len(transposedData))] , predictor([instance])

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

