from mime import *
import pmlb
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt

x, y = pmlb.fetch_data('iris', return_X_y=True)
data = pd.DataFrame(x)
mime = Mime(data, y)
blackBox = SVC()
blackBox.fit(x[:100], y[:100])
all_explanations = []
for i in range(100):
    explanation, pred = mime.explain(x[101], blackBox.predict)
    print(explanation, pred)
    all_explanations.append(explanation)

importance_distributions = [list(column) for column in zip(*all_explanations)]
for feature in importance_distributions:
    plt.hist(feature)
    plt.xlim((0, 5))
    plt.show()
