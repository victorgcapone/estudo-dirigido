from mime import *
import pmlb
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from scipy.spatial.distance import pdist
#import matplotlib.pyplot as plt

x, y = pmlb.fetch_data('spambase', return_X_y=True)
data = pd.DataFrame(x)
explainer = Mime(data, y, categorical=[55, 56])
blackBox = SVC()
blackBox.fit(x[:3000], y[:3000])
#print(blackBox.score(x[3000:], y[3000:]))
all_explanations = []

explanations = []
samples = x[:500]
for instance in samples:
    explanation, pred = explainer.explain(instance, blackBox.predict)
    explanations.append(explanation)

print(explanations)
explanations = np.asarray(explanations)
np.save("explanations", explanations)
np.save("distances", pdist(explanations))
np.save("instances", np.asarray(samples))

count=0
# Separability
for i, e in enumerate(explanation):
    for j, e2 in enumerate(explanations[i+1:]):
        if(np.equals(e,e2).all()):
            count+=1.0

print("Separability: %.4f" % (count/len(explanation)**2))