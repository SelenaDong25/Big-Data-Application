import statsmodels.api as sm
import pandas as pd
import numpy as np


data = pd.read_csv("C:\LWTECH\CSD438-BigData\\rain.csv")
#print(data)

yearSold = data["YrSold"]
salePrice = data["SalePrice"]

 
yearSold = sm.add_constant(yearSold)
model = sm.OLS(salePrice, yearSold)
result = model.fit()
print(result.params)
print(result.summary())