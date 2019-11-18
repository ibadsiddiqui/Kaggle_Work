import pandas as pd

melbourn_file_path = "./train.csv"

melbourn_data = pd.read_csv(melbourn_file_path)
#print(melbourn_data)

y = melbourn_data.SalePrice
features = ["LotArea","YearBuilt" ,"1stFlrSF" ,"2ndFlrSF" , "FullBath","BedroomAbvGr" ,"TotRmsAbvGrd"]
X = melbourn_data[features]
