import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
from mime import *
from lime import lime_tabular
import pmlb
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import spearmanr, pearsonr

# Fetch the data
data = pmlb.fetch_data("churn")
categorical = [0,2,3,4,5]
data = data.sample(frac=1.0)
data_x, data_y = data.drop('target', axis=1), data['target']

train_x = data_x.sample(frac=0.7)
train_y = data_y.reindex(train_x.index)
test_x = data_x.drop(train_x.index)
test_y = data_y.reindex(test_x.index)
test_x.index.intersection(train_x)
params = {
    'criterion' : ['gini', 'entropy'],
    'n_estimators' : [10*c for c in range(1,5)]
}
# Training a SVC
classifier = RandomForestClassifier()
search = GridSearchCV(classifier, param_grid=params, scoring="recall")
search.fit(train_x, train_y)
# Final score is
print(search.best_score_)
print(search.best_estimator_.score(test_x, test_y))
classifier = search.best_estimator_
mime_explainer = Mime(data_x, data_y, categorical)
instance = data_x.sample(1)
importances, prediction = mime_explainer.explain(instance.values[0], classifier.predict)
print(importances, prediction)
lime_explainer = lime_tabular.LimeTabularExplainer(data_x.values, feature_names=data_x.columns,
                                                   class_names=['no-churn', 'churn'],
                                                   categorical_features=categorical)
exp = lime_explainer.explain_instance(instance.values[0], classifier.predict_proba, top_labels=1, num_features=100)

def rank_order(lizt):
    order = [(ind, value) for ind, value in enumerate(lizt)]
    order.sort(key=lambda x:x[1], reverse=True)
    return order

ord_mime = rank_order(importances)
print(ord_mime)
ord_lime = exp.as_map()[0]
print(ord_lime)
# plt.xticks(range(len(ord_mime)))
# plt.yticks(range(len(ord_mime)))
# plt.scatter([v[0] for v in ord_mime], [v[0] for v in exp.as_map()[0]])
# plt.show()
# pearsonr([v[0] for v in ord_mime], [v[0] for v in exp.as_map().popitem()[1]])

def calculate_rho_instance(instance):
    importances, prediction = mime_explainer.explain(instance[0], classifier.predict)
    exp = lime_explainer.explain_instance(instance[0], classifier.predict_proba, top_labels=1, num_features=100)
    ord_mime = rank_order(importances)
    return pearsonr([v[0] for v in ord_mime], [v[0] for v in exp.as_map().popitem()[1]])

data_explanations = data_x.sample(500)
rhos = []
confidences = []

for instance in data_explanations.values:
    instance = np.array([instance])
    rho, confidence = calculate_rho_instance(instance)
    rhos.append(rho)
    confidences.append(confidence)

print(sum(rhos)/len(rhos))