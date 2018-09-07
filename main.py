import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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
#print(df.head())

#Rename value to total
d = {'value':'Total'}
# Pandas group by year then sum
nett_year = df.groupby(['year'], as_index=False)['value'].sum().rename(columns=d)

#Gathering x and y data
x_year = nett_year['year']
y_nett = nett_year['Total']
#Initialise plot
plt.figure(figsize=(12, 9))
ax = plt.subplot(111)


#Convert to array
x_year = np.expand_dims(np.array(x_year), axis=1)
y_nett = np.expand_dims(np.array(y_nett), axis=1)

#plot_nice_lineplot is a plotting function with nicer defaults

#Plot original data
plot.plot_nice_lineplot(x_year, y_nett, title='Yearly manufacturing output, millions/SGD', ax=ax, line_label='Original')

#Plot unregularised linear fit
lr = LinearRegression()
lr.fit(x_year, y_nett)
y_fitted = lr.predict(x_year)
plot.plot_nice_lineplot(x_year, y_fitted, ax=ax, col_index=6, line_label='Linear Fit', linestyle='--')

#Plot unregularised polynomial fit
poly_lr = LinearRegression()

x_norm = x_year - x_year[0]
x_poly = np.concatenate((x_norm, x_norm**2), axis = 1)

poly_lr.fit(x_poly, y_nett)
y_fitted = poly_lr.predict(x_poly)   
print(y_fitted.shape)
plot.plot_nice_lineplot(x_year, y_fitted, ax=ax, col_index=14, line_label='Square Fit', linestyle='--')

#Show graph
plt.show()

