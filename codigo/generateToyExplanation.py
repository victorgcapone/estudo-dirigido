from mime import *
import pmlb
import pandas as pd
from sklearn.svm import SVC

x,y = pmlb.fetch_data('iris', return_X_y=True)
data = pd.DataFrame(x)
mime = Mime(data,y)
blackBox = SVC()
blackBox.fit(x[:100], y[:100])
explanation = mime.explain(x[101], blackBox.predict)
print(explanation)
