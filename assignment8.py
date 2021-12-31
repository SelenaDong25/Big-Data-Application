"""Instructions

This assignment will be graded manually you can ignore the feedback from the auto-grader. 

Create a shared Google Document with your answer.

 

Part 1 (50 points)

For this data set

Kaggle King County Housing Data: https://www.kaggle.com/harlfoxem/housesalesprediction (you may need to create a user to download the data set)

Answer these questions:

(Statistics) What is the average home price in the zip code 98034 and what is the standard deviation.
(Regression) What are the best predictors for home price from the ones in the file? Show the model.
(Decision Tree) What are the best predictors for whether a home has a waterfront? Show the model. 
(Clustering) Cluster the data using this columns: bedrooms, bathrooms, sqft_living, floors, waterfront, price. Name the clusters.
(Forecasting) What is the expected average home price for January 2016 based on the average home prices from previous months?
Each answer and script with worth 10 points.

Part 2 (20 points)

Using the same data set design and answer two questions of your choosing (new questions). Each answer and script is worth 10 points.

Submission Format

The document should include:

- An introduction on the data set including information on what it represented (i.e. relevant columns and their meaning) - 10 points.

- The answers to all the questions above and the scripts - 70 points

- A conclusion summarizing your findings - 20 points
"""
import statsmodels.api as sm
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.preprocessing as pre
import sklearn.tree as tree
from sklearn import metrics
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering

#import plotly.express as px

#from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

 
data = pd.read_csv("C:\LWTECH\CSD438-BigData\\final\kc_house_data.csv")

######################################################################

# (Statistics) What is the average home price in the zip code 98034 and what is the standard deviation.

kirkland = data.loc[data["zipcode"]== 98034]

price = kirkland["price"]
#p_mean = np.mean(price)
p_average = sum(price)/len(price)
p_sd = np.std(price)
#print(p_mean)
print("Average home price in 98034 is:", p_average)
print("Zip code 98034 home price standard deviation is: ", p_sd)

####################################################################

# (Forecasting) What is the expected average home price for January 2016 based on the average home prices from previous months?

avg_all = data.groupby([data["date"].str[:6]], as_index=False).agg(avg_price=pd.NamedAgg(column="price", aggfunc="mean"))

avg_all["Month"] = [1,2,3,4,5,6,7,8,9,10,11,12,13]
print( "Previous monthly average sale price list is: \n" , avg_all)

month_avg = avg_all["avg_price"]
month = avg_all["Month"]

month = sm.add_constant(month)
month_model = sm.OLS(month_avg, month)
month_fit = month_model.fit()

# month.loc[13] = [1,14]  #2015/6
# month.loc[14] = [1,15]  #2015/7
# month.loc[15] = [1,16]  #2015/8
# month.loc[16] = [1,17]  #2015/9
# month.loc[17] = [1,18]  #2015/10
# month.loc[18] = [1,19]  #2015/11
# month.loc[19] = [1,20]  #2015/12
month.loc[20] = [1,21]  #2016/1
month_new = month.loc[20:20]
predictions = month_fit.predict(month_new)
prediction = predictions.values
print("Predicted monthly average sale price for Jan 2016 is: ",  prediction)
"""
#############################################################################
#(Decision Tree) What are the best predictors for whether a home has a waterfront? Show the model. 

# from PIL import Image

n = len(data)
p = int(n * 0.95)
train = data[0:p]
test = data[(p+1):n-1]
#print(train)

X_train = train[["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "view", 
                "condition", "grade",    "sqft_living15", "sqft_lot15"]].values

Y_train = train['waterfront'].values

X_test = test[["price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "view", 
                "condition", "grade",   "sqft_living15", "sqft_lot15"]].values

Y_test = test['waterfront'].values

classifier = tree.DecisionTreeClassifier(max_depth=5)
my_tree = classifier.fit(X_train, Y_train)

Y_predict = my_tree.predict(X_test)
print(Y_predict)
print("Accuracy:",metrics.accuracy_score(Y_test, Y_predict))

out = open('C:\LWTECH\CSD438-BigData\\tree3.dot', 'w')
dot_output = tree.export_graphviz(my_tree, out_file=out, feature_names=[ "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "view", 
                "condition", "grade",  "sqft_living15", "sqft_lot15"], 
                 class_names=[ 'Waterfront', 'NotWaterfront'])

print(my_tree.predict([[1175000, 2, 2.5, 1770, 7155, 2, 5, 3, 8, 2410, 10476]]))
print(my_tree.predict([[1335000,	3,	2,	1410,	44866,	1, 6,	4,	7,  2950,	29152]]))
print(my_tree.predict([[835000,	2,	2,	1410,	44866,	2, 2,	1,	2,  1000,	20000]]))

#################################################################################
#(Clustering) Cluster the data using this columns: bedrooms, bathrooms, sqft_living, floors, waterfront, price. Name the clusters.

df = data[['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'price']]
#print(df)
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
cluster.fit(df.values)
 
print(cluster.labels_)

cluster0 = df[cluster.labels_==0]
cluster1 = df[cluster.labels_==1]
cluster2 = df[cluster.labels_==2]
cluster3 = df[cluster.labels_==3]

 
import statistics as st
 
print("Luxury Expensive Home with Great View:")
for i in range(6):
    print(str(round(st.mean(cluster0.iloc[:,i]), 2)) + ": " + cluster0.columns[i])
 
print("\n")
print("Good Sized Family Home:")
for i in range(6):
    print(str(round(st.mean(cluster1.iloc[:,i]), 2)) + ": " + cluster0.columns[i])
print("\n") 
print("Basic Living Home:")
for i in range(6):
    print(str(round(st.mean(cluster2.iloc[:,i]), 2)) + ": " + cluster0.columns[i])
print("\n") 
print("Executive Home:")
for i in range(6):
    print(str(round(st.mean(cluster3.iloc[:,i]), 2)) + ": " + cluster0.columns[i])
print("\n") 


#################################################################################
#(Regression) What are the best predictors for home price from the ones in the file? Show the model.
price = data["price"]
bedrooms = data["bedrooms"]
bathrooms = data["bathrooms"]
sqft_living = data["sqft_living"]
sqft_lot = data["sqft_lot"]
floors = data["floors"]
waterfront = data["waterfront"]
condition = data["condition"]
yr_built = data["yr_built"]
view = data["view"]
grade = data["grade"]
yr_renovated = data["yr_renovated"]

df = pd.DataFrame({"bedrooms" : bedrooms, "grade": grade,
                   "bathrooms" : bathrooms,  "sqft_living" : sqft_living,
                    "sqft_lot" : sqft_lot, "waterfront": waterfront,
                    "view": view, "condition": condition,
                   "yr_built": yr_built, "yr_renovated": yr_renovated                  
                    })

Y = price
X = df
#X = sm.add_constant(X)
 
model = sm.OLS(Y, X)
result = model.fit()
 
print(result.summary())

###############################################################################
# plot cluster of price vs zipcode
#df = data[['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'price']]
df1 = data[["zipcode","price"]]
#print(df1)

import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Price Dendograms")
dend = shc.dendrogram(shc.linkage(df1, method='ward'))

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(df1)

plt.figure(figsize=(10, 7))
plt.scatter(df1.iloc[:,0:1], df1.iloc[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()
"""
