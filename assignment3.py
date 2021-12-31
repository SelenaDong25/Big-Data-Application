import pandas as pd
import statsmodels.api as sm
import numpy as np

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from matplotlib import pyplot

# Read the data file sales_quarter.csv. When using your computer

# modify this path to match your local path

data = pd.read_csv("C:\LWTECH\CSD438-BigData\sales_quarter.csv")

# Calculate the Partial Auto-Correlation in the data in the sales column

# When submitting your work don't show the plot (i.e don't run pyplot.show())


# Your code starts after this line

plot_pacf(data['sales'])
pyplot.show()
# Your code ends before this line


# Use the lags from the previous part to create a linear model that predicts 

# future sales based on past sales

# Name the shifted variables s1, s2, ..., sn, where s1 is the one with the smallest lag

# s2 the one with the second smallest and so on.

# Summarize the model and use it to predict the sales for the next time period in the set.

# # Your code starts after this line
sales = data["sales"]
quarter = data["Quarter"]
s1 = sales.shift(periods=1)
s2 = sales.shift(periods=2)
s11 = sales.shift(periods=11)

predictor = pd.DataFrame({'s1':s1, 's2':s2})
#print(sales_shift)



Y = sales[11:]
X = predictor[11:]
# #X = sm.add_constant(X)
 
m = sm.OLS(Y, X)
print(m.fit().summary())

t = sales.size
input = pd.DataFrame({'s1':[sales[t-1]], 's2':[sales[t-2]]})
print(m.fit().predict(input))

# Your code ends before this line

# Use the model to predict the sales in the next quarter rounded to two decimal places


# Your code starts after this line

qt_data = data.groupby([data["Quarter"].astype(int)], as_index=False).agg(qt_sale=pd.NamedAgg(column="sales", aggfunc="sum"))

qt_data["Qt"] = [1,2,3,4,5,6,7,8,9,10,11,12]
print(qt_data)

qt_sale = qt_data["qt_sale"]
qt = qt_data["Qt"]
#quarter_sale = quarter_data["sales"]
#quarter = quarter_data[indexer]
# tt = quarter_data
# print(tt)
# plot_acf(quarter_sale)
# pyplot.show()

qt = sm.add_constant(qt)
qt_m = sm.OLS(qt_sale, qt)
qt_fit = qt_m.fit()

qt.loc[12] = [1,13]
qtnew = qt.loc[12:12]
predictions = qt_fit.predict(qtnew)
prediction = predictions.values
print(prediction)

#print(m.fit().predict(input))
# print(test)
#print(test.groupby(test["Quarter"]).sum())

#data.groupby(data.Quarter.str[:1])['sales'].sum()

# Your code ends before this line