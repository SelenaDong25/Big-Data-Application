import statsmodels.api as sm
import numpy as np
import pandas as pd

m00 = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
m01 = [0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
m02 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
m10 = [0, 0, 0, 1, 1, 1, 0, 1, 1, 1]
m11 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
m12 = [1, 1, 1, 1, 0, 0, 1, 1, 1, 1]
m20 = [0, 1, 0, 1, 1, 1, 0, 1, 1, 1]
m21 = [0, 1, 1, 1, 1, 1, 0, 1, 1, 0]
m22 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
m30 = [0, 1, 0, 0, 0, 1, 0, 1, 0, 1]
m31 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
m32 = [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]
m40 = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
m41 = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
m42 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
 
# df = pd.DataFrame({"m00" : m00, "m01" : m01, "m02" : m02,
#                    "m10" : m10, "m11" : m11, "m12" : m12,
#                    "m20" : m20, "m21" : m21, "m22" : m22,
#                    "m30" : m30, "m31" : m31, "m32" : m32,
#                    "m40" : m40, "m41" : m41, "m42" : m42,
#                     })
df = pd.DataFrame({"m00" : m00, "m01" : m01, 
                   "m10" : m10,  "m12" : m12,
                    "m21" : m21, 
                    
                   "m40" : m40
                    })


Y = values
X = df
#X = sm.add_constant(X)
 
model = sm.OLS(Y, X)
result = model.fit()
 
print(result.summary())