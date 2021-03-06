from mime import *
import pmlb
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt

x, y = pmlb.fetch_data('spambase', return_X_y=True)
data = pd.DataFrame(x)
explainer = Mime(data, y, categorical=[55, 56])
blackBox = SVC()
blackBox.fit(x[:3000], y[:3000])
print(blackBox.score(x[3000:], y[3000:]))
all_explanations = []

for i in range(100):
    explanation, pred = explainer.explain(x[-1], blackBox.predict)
    all_explanations.append(explanation)

print(all_explanations[:10])
importance_distributions = [list(column) for column in zip(*all_explanations)]
for feature in importance_distributions:
    plt.hist(feature)
    plt.xlim((0, 5))
    plt.show()
