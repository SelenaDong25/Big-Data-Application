import statsmodels.api as sm
import numpy as np
import pandas as pd
 
data = pd.read_csv("C:\LWTECH\CSD438-BigData\kc.csv")
 
sale_price = data["SalePrice"]
live_area = data["GrLivArea"]

live_area = sm.add_constant(live_area)
model = sm.OLS(sale_price, live_area)
result = model.fit()
print(result.params)
print(result.summary())
 
# min_height = min(heights)
# max_height = max(heights)
 
# print("Min Height: " + str(min_height))
# print("Max Height: " + str(max_height))
 
# median_height = st.median(heights)
# print("Median Height: " + str(median_height))




#df = pd.read_csv('data/kc.csv')