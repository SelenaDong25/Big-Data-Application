import pandas as pd
import statistics as st
import numpy as np
from scipy import stats

data = pd.read_csv("C:\LWTECH\CSD438-BigData\weight-height-v2.csv")
 
heights = data["Height"]
#Calculate the min, max, median, average, and 10th percentile Height
h_mean = np.mean(heights)
h_sd = np.std(heights)

min_height = min(heights)
max_height = max(heights)
median_height = st.median(heights)
average_height = sum(heights)/len(heights)
percentile_height = np.percentile(heights,10)



print("Min Height: " + str(round(min_height, 2)))
print("Max Height: " + str(round(max_height, 2)))
print("Average Height: " + str(round(average_height, 2)))
print("Median Height: " + str(round(median_height, 2)))
print("10th Percentile Height: " + str(round(percentile_height, 2)))

print("Mean: " + str(round(h_mean, 2)))
print("Standard deviation: " + str(round(h_sd, 2)))


#Assuming that the data is normally distributed answer the following questions

# Q1 - What is the minimum height for a door that allows 83% of 

#      the people to go through without bending?

# Q2 - What is the minimum height for a door that allows 95% of 

#      the people to go through without bending?

# Q3 - What percentage of people are taller than 66 inches?

# Q4 - What percentage of people are shorter than 66 inches?

# Q5 - What percentage of people are between 60 and 70 inches?


# Your code starts after this line
a1 = stats.norm.ppf(0.83,loc=h_mean, scale=h_sd)
a2 = stats.norm.ppf(0.95,loc=h_mean, scale=h_sd)

a3 = stats.norm.cdf(max_height, loc=h_mean, scale=h_sd) - stats.norm.cdf(66, loc=h_mean, scale=h_sd)
a4 = stats.norm.cdf(66, loc=h_mean, scale=h_sd)
a5 = stats.norm.cdf(70, loc=h_mean, scale=h_sd) - stats.norm.cdf(60, loc=h_mean, scale=h_sd)
# Your code ends before this line


print("Q1: " + str(round(a1, 2)))
print("Q2: " + str(round(a2, 2)))
print("Q3: " + str(round(a3, 2)))
print("Q4: " + str(round(a4, 2)))
print("Q5: " + str(round(a5, 2)))