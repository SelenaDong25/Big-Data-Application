from pandas import read_csv
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

data = read_csv("C:\LWTECH\CSD438-BigData\\ac_data.csv")

plot_pacf(data['x5'])
pyplot.show()
