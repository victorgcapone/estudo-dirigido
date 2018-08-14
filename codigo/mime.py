# coding:utf-8
import math
from numpy import digitize
# MIME is a Mutual Information Model-Agnostic Explanator for machine learning
# black boxes, it uses a similar aproach to that of LIME, probing the decision
# with perturbed versions of the instance being explained

# Used to encapsulate all data and metadata for message passing between layers in mime
class DataWrapper(object):

    def __init__(data, categorical):
        self.data = data
        self.categorical = categorical


class Mime(object):

    # Mime works with pandas DataFrames
    # Parameters:
    # dataframe    : your data
    # categorical  : a list with the index of the categorical columns
    # preprocessor : the preprocessor for your explanations (default: MimePreprocessor)
    # explainer    : the explainer that will generate your explanations (default: MimeExplainer)
    # parameters   : the parameters (if any) you want to pass to your preprocessor or explainer, should be a dict
    def __init__(self, dataframe, categorical, preprocessor=MimePreprocessor, explainer=MimeExplainer, preprocessorParameters={}, explainerParameters={}):
        self.data = DataWrapper(dataframe, categorical)
        self.preprocessor = preprocessor(data, **preprocessorParameters)
        self.explainer = explainer(**explainerParameters)
        self.preprocessor.preprocess()

    # Parameters:
    # instance  : the instance being explained
    # predictor : a function that takes instance and returns a black-box prediction
    def explain(self, instance, predictor):
        self.explainer(instance, predictor, **self.preprocessor.computed)

# Before explaining our instances we may need to do
# some pre-calculations for some reason, this is
# the role of the preprocessor
class MimePreprocessor(object):

    # You may use kwargs to set some parameters for the preprocessor
    def __init__(self, data, **kwargs):
        self.data = data
        self.args = kwargs

    def preprocess(self):
        # For MIME we need to precompute the optimal number of bins for each non-categorical feature
        # We do this using Sturge's Formula and the Sample Size
        # Then we change our data to the binned version
        self.computed = {}
        self.computed["optimalBins"] = math.log(data.data.shape[0], 2) + 1
        self.computed["sampleSize"] = 0.1 * data.data.shape[0] # 10% of the data size
        for i in range(data.data.shape[1]):
            if i in data.categorical:
                continue
           tmp = self.data.data.T.values


# At last, the explainer takes an instance and pre-computed data
# and generates an explanation for it
class MimeExplainer(object):

    class __init__(self, **kwargs):
       self.args = kwargs

    # You can use kwargs to pass parameters precomputed by the preprocessor
    # MimeExplainer expects an "optimalBins" parameters and a "sampleSize"
    # parameter
    class explain(self, instance, predictors, **kwarg):
        pass
