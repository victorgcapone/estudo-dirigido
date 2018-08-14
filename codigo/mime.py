# coding:utf-8

# MIME is a Mutual Information Model-Agnostic Explanator for machine learning
# black boxes, it uses a similar aproach to that of LIME, probing the decision
# with perturbed versions of the instance being explained

class Mime(object):

    # Mime works with pandas DataFrames
    def __init__(self, dataframe, preprocessor=MimePreprocessor, explainer=MimeExplainer):
        self.data = dataframe
        self.preprocessor = preprocessor(data)

    def explain(self, instance):
        pass

# Beforing explaining our instances we may need to do
# some pre-calculations for some reason, this is
# the role of the preprocessor
class MimePreprocessor(object):

    # You may use kwargs to set some parameters for the preprocessor
    def __init__(self, data, **kwargs):
        self.data = data
        self.args = kwargs

    def preprocess(self, **kwargs):
        pass

# At last, the explainer takes an instance and pre-computed data
# and generates an explanation for it
class MimeExplainer(object):

    class __init__(self):
        pass

    # You can use kwargs to pass parameters precomputed by the preprocessor
    class explain(self, instance, **kwarg):
        pass