import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('Churn_Modelling.csv', index_col = 0)
data = data.dropna()

x = data.iloc[ : , :-1]
y = data["Exited"]
x = pd.get_dummies(x, drop_first = True)
x = x.dropna()
sc = StandardScaler()
sc.fit(x)
numeric = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
x[numeric] = sc.fit_transform(x[numeric])
rfc = RandomForestClassifier(max_depth = 2, random_state = 0)
rfc.fit(x, y)

lgr = LogisticRegression(random_state = 0, solver = "liblinear")
lgr.fit(x, y)
print(rfc.score(x, y))
print(lgr.score(x, y))




