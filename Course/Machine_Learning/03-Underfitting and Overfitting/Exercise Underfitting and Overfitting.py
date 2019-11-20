# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:58:53 2019

@author: ibad.siddiqui
"""
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


melbourn_file_path = "./../train.csv"

melbourn_data = pd.read_csv(melbourn_file_path)

y = melbourn_data.SalePrice

features = ["LotArea","YearBuilt" ,"1stFlrSF" ,"2ndFlrSF" , "FullBath","BedroomAbvGr" ,"TotRmsAbvGrd"]
X = melbourn_data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
    

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}

best_leaf_size = min(scores, key=scores.get)

final_model = DecisionTreeRegressor(max_depth=best_leaf_size)
final_model.fit(train_X, train_y)
print(final_model.predict(val_X))