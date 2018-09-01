# coding:utf-8
import math
from numpy import digitize, linspace
import random
import pandas as pd


def prob(e, space):
    p = float(space.count(e))/len(space)
    return p


def weighted_mutual_information(x, y, weights):
    x = list(x)
    y = list(y)
    if len(weights) != len(x[0]):
        raise ValueError("There must be as many features as there are weights sets")
    xt = [list(column) for column in zip(*x)]
    uy = set(y)
    mi = [0] * len(xt)
    for index, column in enumerate(xt):
        joint = list(zip(y, column))
        for vY in uy:
            for i, vX in enumerate(column):
                w = weights[index][vX] 
                p = prob((vY, vX), joint)
                prob_ratio = float(p)/(prob(vX, column)*prob(vY, y))
                if p > 0:
                    mi[index] += w * p * math.log(prob_ratio)
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
        self.compute_params(self.data)

    def compute_params(self, data):
        self.computed["optimalBins"] = int(math.log(data.data.shape[0], 2) + 1)
        self.computed["sampleSize"] = int(0.1 * data.data.shape[0])
        self.computed["bins"] = []
        tmp = data.data.T.values
        for i in range(data.data.shape[1]):
            if i in data.categorical:
                continue
            # Digitizes the values on the given columns
            bins = linspace(min(tmp[i]), max(tmp[i]), self.computed["optimalBins"])
            self.computed["bins"].append(bins)

    def preprocess(self, data=None):
        if data is not None:
            working_data = data
            self.compute_params(working_data)
        # For MIME we need to precompute the optimal number of bins for each non-categorical feature
        # We do this using Sturge's Formula and the Sample Size
        # Then we change our data to the binned version
        data_copy = working_data.data.copy()
        tmp = working_data.data.T.values
        for i in range(data.data.shape[1]):
            if i in working_data.categorical:
                continue
            binned = digitize(tmp[i], self.computed["bins"][i])
            data_copy[data_copy.columns[i]] = binned
        new_data = DataWrapper(data_copy, working_data.target, working_data.categorical)
        return new_data


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
        preprocessor.compute_params(data)
        # Generate neighborhood samples
        neighborhood = pd.DataFrame(self.sampler.sample_neighborhood(instance, data,
                                                                     preprocessor.computed['sampleSize']))
        # Label the samples
        neighborhood_labels = predictor(neighborhood)
        # Weight the samples
        bins_weights = self.weight_bins(instance, preprocessor.computed["bins"])
        neighborhood_data = DataWrapper(neighborhood, neighborhood_labels, data.categorical)
        neighborhood_data = preprocessor.preprocess(neighborhood_data)
        # For each feature, calculates its mutual information with the labels
        return weighted_mutual_information(neighborhood_data.data.values,
                                           neighborhood_labels, bins_weights), predictor([instance])

    def weight_bins(self, instance, bins):
        w = []
        width = 10
        for f in range(len(instance)):
            bins_w = []
            bins_width = bins[0][1]-bins[0][0]
            for b in range(len(bins[0])):
                # Falta, gerar a última bin
                bin_midpoint = bins[f][b]-bins_width/2
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
                 preprocessor_parameters={}, explainer_parameters={}):
        self.data = DataWrapper(dataframe, target, categorical)
        self.preprocessor = preprocessor(self.data, **preprocessor_parameters)
        self.explainer = explainer(**explainer_parameters)

    # Parameters:
    # instance  : the instance being explained
    # predictor : a function that takes instance and returns a black-box prediction
    def explain(self, instance, predictor):
        return self.explainer.explain(instance, self.data, predictor, self.preprocessor)

