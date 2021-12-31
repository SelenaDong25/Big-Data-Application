import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.preprocessing as pre
import sklearn.tree as tree
from PIL import Image

data = pd.read_csv("C:\LWTECH\CSD438-BigData\WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Prepare the data by encoding the revelant columns


# Your code starts after this line
label_encoder = pre.LabelEncoder()
label_encoder.fit(data['Attrition'].astype(str))
data['Attrition'] = label_encoder.transform(data['Attrition'].astype(str))

label_encoder = pre.LabelEncoder()
label_encoder.fit(data['BusinessTravel'].astype(str))
data['BusinessTravel'] = label_encoder.transform(data['BusinessTravel'].astype(str))

label_encoder = pre.LabelEncoder()
label_encoder.fit(data['Department'].astype(str))
data['Department'] = label_encoder.transform(data['Department'].astype(str))

label_encoder = pre.LabelEncoder()
label_encoder.fit(data['EducationField'].astype(str))
data['EducationField'] = label_encoder.transform(data['EducationField'].astype(str))

label_encoder = pre.LabelEncoder()
label_encoder.fit(data['Gender'].astype(str))
data['Gender'] = label_encoder.transform(data['Gender'].astype(str))

label_encoder = pre.LabelEncoder()
label_encoder.fit(data['JobRole'].astype(str))
data['JobRole'] = label_encoder.transform(data['JobRole'].astype(str))

label_encoder = pre.LabelEncoder()
label_encoder.fit(data['MaritalStatus'].astype(str))
data['MaritalStatus'] = label_encoder.transform(data['MaritalStatus'].astype(str))

label_encoder = pre.LabelEncoder()
label_encoder.fit(data['Over18'].astype(str))
data['Over18'] = label_encoder.transform(data['Over18'].astype(str))

label_encoder = pre.LabelEncoder()
label_encoder.fit(data['OverTime'].astype(str))
data['OverTime'] = label_encoder.transform(data['OverTime'].astype(str))




#print(X)

# Your code ends before this line


# This section separates the data between training and testing

# don't make any changes to it


n = len(data)
p = int(n * 0.95)
train = data[0:p]
test = data[(p+1):n-1]
print(train)
# Create a decision three with max_depth = 5 to predict attrition

# based on all the columns in the train dataset


# Your code starts after this line

X_train = train[[ 'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 
        'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 
        'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']].values
Y_train = train['Attrition'].values

X_test = test[[ 'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 
        'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 
        'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']].values

classifier = tree.DecisionTreeClassifier(max_depth=5)
my_tree = classifier.fit(X_train, Y_train)

Y_predict = my_tree.predict(X_test)
print(Y_predict)

# out = open('C:\LWTECH\CSD438-BigData\\tree2.dot', 'w')
# dot_output = tree.export_graphviz(my_tree, out_file=out, feature_names=[ 'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 
#         'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 
#         'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
#         'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'],

#     class_names=[ 'Yes', 'No'])


# Your code ends before this line


# Use your tree to predict attrition for the values in the test data set

# Print the predictions


# Your code starts after this line
#print(my_tree.predict([Y]))

# Your code ends before this line