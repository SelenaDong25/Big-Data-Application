import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

data = pd.read_csv('C:\LWTECH\CSD438-BigData\Ingredients.csv')

# Your code starts after this line
# The file Ingredients.csv contains the ice cream flavors and their ingredients for the store.

# Flavor codes are a linear model of the ingredients.

# Use the data to calculate the flavor code for a new flavor with this combination of ingredients:

# [47, 31, 48, 18, 49, 43]

# Sales seem to also be a linear function of the ingredients. Use the data to predict the expected sales for this new flavor.

# Your code should print the summary of the linear models and the predictions. Round the predictions to two decimal places.

# Your code ends before this line
values = data["Flavor"]
m01 = data["Ingredient 1"]
m02 = data["Ingredient 2"]
m03 = data["Ingredient 3"]
m04 = data["Ingredient 4"]
m05 = data["Ingredient 5"]
m06 = data["Ingredient 6"]

df = pd.DataFrame({"m01" : m01, "m02" : m02, 
                   "m03" : m03,  "m04" : m04,
                   "m05" : m05,  "m06" : m06
                    })


Y = values
X = df
#X = sm.add_constant(X)
 
model = sm.OLS(Y, X)
result = model.fit()
 
print(result.summary())

flavor_code = 47*1.0 + 31*1.0 + 48*1.0 + 18*1.0 + 49*1.0 + 43*1.0
print(flavor_code)

#plot_pacf(data['Sales'], lags=9)
#plt.show()
sales = data["Sales"]
# s1 = sales.shift(periods=1)
# s2 = sales.shift(periods=2)
# s3 = sales.shift(periods=3)
# s4 = sales.shift(periods=4)
# s5 = sales.shift(periods=5)
# s6 = sales.shift(periods=6)
# s7 = sales.shift(periods=7)
# s8 = sales.shift(periods=8)
# s9 = sales.shift(periods=9)


#predictor = pd.DataFrame({'s1':s1, 's2':s2,   's7':s7,  's9':s9})
#predictor = pd.DataFrame({'s1':s1, 's3':s3, 's4':s4, 's5':s5, 's6':s6, 's8':s8, 's11':s11, 's18':s18})
# Y1 = sales[9:]
# X1 = predictor[9:]
#X = sm.add_constant(X)
 
# m = sm.OLS(Y1, X1)
# print(m.fit().summary())

# t = sales.size
#input = pd.DataFrame({'s1':[sales[t-1]], 's4':[sales[t-4]],  's5':[sales[t-5]], 's8':[sales[t-8]]})
# input = pd.DataFrame({'s1':[sales[t-1]], 's3':[sales[t-3]], 's4':[sales[t-4]], 's5':[sales[t-5]], 's6':[sales[t-6]], 's8':[sales[t-8]], 's11':[sales[t-11]], 's18':[sales[t-18]]})
# print(round(m.fit().predict(input), 2))
# df1 = pd.DataFrame({"m01" : m01, "m02" : m02, 
#                     "m03" : m03,  "m04" : m04,
#                     "m05" : m05,  "m06" : m06
#                     })

df1 = pd.DataFrame({"m01" : m01, "m02" : m02, 
                     "m04" : m04,
                     "m06" : m06
                    })

Y1 = sales
X1 = df1
X1 = sm.add_constant(X1)
model1 = sm.OLS(Y1, X1)
result1 = model1.fit()
 
print(result1.summary())

#t = sales.size
#input = pd.DataFrame({'m01':[sales[t-1]], 'm02':[sales[t-2]],  'm04':[sales[t-4]], 'm06':[sales[t-6]]})
# input = pd.DataFrame({'s1':[sales[t-1]], 's3':[sales[t-3]], 's4':[sales[t-4]], 's5':[sales[t-5]], 's6':[sales[t-6]], 's8':[sales[t-8]], 's11':[sales[t-11]], 's18':[sales[t-18]]})
#print(round(model1.fit().predict(input), 2))

predict = 65.2397 + 313.9303*47 - 1.0211*31 +679.7972*18 + 908.7737*43
print(round(predict, 2))

