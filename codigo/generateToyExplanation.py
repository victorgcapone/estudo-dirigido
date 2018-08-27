from mime import *
import pmlb
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import mutual_info_score

x,y = pmlb.fetch_data('iris', return_X_y=True)
data = pd.DataFrame(x)
mime = Mime(data,y)
blackBox = SVC()
blackBox.fit(x[:100], y[:100])
#print([mutual_info_score(data.T.values[i], y) for i in range(4)])
for i in range(101, 115):
    explanation = mime.explain(x[i], blackBox.predict)
    print(explanation)
