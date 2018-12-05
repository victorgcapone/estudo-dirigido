from mime import *
import pmlb
import pandas as pd
from sklearn.svm import SVC
from scipy.spatial.distance import pdist
import numpy as np
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

#print(explanations)
distances = pdist(explanation)
print(distances)
np.save("distances", distances)
importance_distributions = [list(column) for column in zip(*all_explanations)]
"""for feature in importance_distributions:
    plt.hist(feature)
    plt.xlim((0, 5))
    plt.show()"""