import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf

data = pd.read_csv("C:\LWTECH\CSD438-BigData\Quarter_Sales.csv")
# The file Quarter_Sales.csv contains ice cream sales data for each quarter from 2006 till 2020.

# Using auto-correlation techniques, predict the sales for Q1 of 2021.

# Your code should print the summary of the linear model and the prediction
# Your code starts after this line
#plot_pacf(data['Sales'])
plot_pacf(data['Sales'])
#plt.show()

sales = data["Sales"]

#quarter = data["Quarter"]
s1 = sales.shift(periods=1)
s3 = sales.shift(periods=3)
s4 = sales.shift(periods=4)
s5 = sales.shift(periods=5)
s6 = sales.shift(periods=6)
s8 = sales.shift(periods=8)
s11 = sales.shift(periods=11)
s18 = sales.shift(periods=18)

predictor = pd.DataFrame({'s1':s1,  's4':s4, 's5':s5, 's8':s8})
#predictor = pd.DataFrame({'s1':s1, 's3':s3, 's4':s4, 's5':s5, 's6':s6, 's8':s8, 's11':s11, 's18':s18})
Y = sales[18:]
X = predictor[18:]
#X = sm.add_constant(X)
 
m = sm.OLS(Y, X)
print(m.fit().summary())

t = sales.size
input = pd.DataFrame({'s1':[sales[t-1]], 's4':[sales[t-4]],  's5':[sales[t-5]], 's8':[sales[t-8]]})
#input = pd.DataFrame({'s1':[sales[t-1]], 's3':[sales[t-3]], 's4':[sales[t-4]], 's5':[sales[t-5]], 's6':[sales[t-6]], 's8':[sales[t-8]], 's11':[sales[t-11]], 's18':[sales[t-18]]})
print(round(m.fit().predict(input), 2))
# Your code ends before this line