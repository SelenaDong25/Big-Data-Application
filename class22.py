import pandas as pd
import statistics as st
import numpy as np
 
data = pd.read_csv("C:\LWTECH\CSD438-BigData/weight-height.csv")
 
heights = data["Height"]
 
min_height = min(heights)
max_height = max(heights)
 
print("Min Height: " + str(min_height))
print("Max Height: " + str(max_height))
 
median_height = st.median(heights)
print("Median Height: " + str(median_height))
 

