import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import os
from matplotlib import pyplot as plt
from util import plot

#Dataset from https://data.gov.sg/dataset/total-output-manufacturing-annual
#Download to data/ and rename to total-output.csv
data_loc = 'data'
file_name = 'total-output.csv'

#Read in data
file_loc = os.path.join(data_loc, file_name)
df = pd.read_csv(file_loc)

# Print file head, data format with years, level 1/2, industry and value
print(df.head())

d = {'year':'Year', 'value':'Total / million SGD'}
# Pandas group by year then sum
nett_year = df.groupby(['year'], as_index=False)['value'].sum().rename(columns=d)

#Gathering x and y data
x_year = nett_year['Year']
y_nett = nett_year['Total / million SGD']
#Initialise plot
ax = plt.subplot(111)
#Plotting function with nicer defaults
plot.plot_nice_lineplot(x_year, y_nett, title='Yearly manufacturing output, millions/SGD', ax=ax)

plt.show()

