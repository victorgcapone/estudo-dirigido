from mime import *
import pmlb
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt

x, y = pmlb.fetch_data('monk1', return_X_y=True)
data = pd.DataFrame(x)
mime = Mime(data, y, categorical=[0, 1, 2, 3, 4, 5])
blackBox = SVC()
blackBox.fit(x[:450], y[:450])
print(blackBox.score(x[450:], y[450:]))
all_explanations = []
for i in range(100):
    explanation, pred = mime.explain(x[301], blackBox.predict)
    all_explanations.append(explanation)

print(all_explanations[:10])
importance_distributions = [list(column) for column in zip(*all_explanations)]
for feature in importance_distributions:
    plt.hist(feature)
    plt.xlim((0, 5))
    plt.show()
