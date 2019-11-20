import pandas as pd
from sklearn.tree import DecisionTreeRegressor


melbourn_file_path = "./../train.csv"

melbourn_data = pd.read_csv(melbourn_file_path)

y = melbourn_data.SalePrice

features = ["LotArea","YearBuilt" ,"1stFlrSF" ,"2ndFlrSF" , "FullBath","BedroomAbvGr" ,"TotRmsAbvGrd"]
X = melbourn_data[features]

melbourn_data_model = DecisionTreeRegressor(random_state=1)
melbourn_data_model.fit(X,y)

predictions = melbourn_data_model.predict(X)
print(predictions)