# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:05:21 2019

@author: ibad.siddiqui
"""


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
house_data = pd.read_csv('../train.csv')
y = house_data.SalePrice
features = ["LotArea","YearBuilt" ,"1stFlrSF" ,"2ndFlrSF" , "FullBath","BedroomAbvGr" ,"TotRmsAbvGrd"]
X = house_data[features]

train_X, val_X, train_y, val_y = train_test_split(X,y, random_state=1)

house_model = DecisionTreeRegressor(random_state=1)
house_model.fit(train_X,train_y)

val_predictions = house_model.predict(val_X)
print(mean_absolute_error(val_predictions,val_y))