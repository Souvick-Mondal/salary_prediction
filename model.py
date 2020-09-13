import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset= pd.read_csv('hiring.csv')

x=dataset.iloc[:, :3]
y=dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
#predictions= regressor.predict(x) 

regressor.fit(x, y)
#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

pickle.dump(regressor,open('model.pkl','wb'))

"""
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
"""
