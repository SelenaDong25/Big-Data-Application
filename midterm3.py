import pandas as pd
import statistics as st
import numpy as np
from scipy import stats

data = pd.read_csv("C:\LWTECH\CSD438-BigData\\ages.csv")

# The file ages.csv contains the ages of 100 random customers that came to the store during the past year.

# Assuming the ages are normally distributed, calculate the probability that a person older than 40 will come to the store. 
# Print the result rounded to two decimals.
# Your code starts after this line
age = data["Age"]
age_mean = np.mean(age)
age_sd = np.std(age)
max_age = max(age)

result = stats.norm.cdf(max_age, loc=age_mean, scale=age_sd) - stats.norm.cdf(40, loc=age_mean, scale=age_sd)
print(round(result, 2))


# Your code ends before this line