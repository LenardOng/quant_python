import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import os
from matplotlib import pyplot as plt
from util import plot
#plot_nice_lineplot is a plotting function with nicer defaults

#Number of level_2 industries to plot
n_ind = 21
#Limit colormesh
vmin_ = -75
vmax_ = 75

#----- DATASET LOADING -----#

#Dataset from https://data.gov.sg/dataset/total-output-manufacturing-annual
#Download to data/ and rename to total-output.csv
data_loc = 'data'
file_name = 'total-output.csv'

#Read in data
file_loc = os.path.join(data_loc, file_name)
df = pd.read_csv(file_loc)

# Print file head, data format with years, level 1/2, industry and value
#print(df.head())


#----- DATASET PREPROCESSING  -----#

#Rename value to total
d = {'value':'Total'}
# Pandas group by year then sum
nett_year = df.groupby(['year'], as_index=False)['value'].sum().rename(columns=d)

# Select n largest indices only (make plot clearer)
largest_n_ind = (df[df['year']==2016]).nlargest(n_ind, 'value')

# Pandas group by year and level_2 industry
industries = largest_n_ind['level_2'].unique()
x_year = nett_year['year']
print('Total number of level_2 industries is {}'.format((df['level_2'].unique()).shape[0]))
print('Plotting the {} largest industries'.format(n_ind))



#Gathering x and y data
x_year = nett_year['year']
y_nett = nett_year['Total']

#Convert to array
x_year = np.expand_dims(np.array(x_year), axis=1)
y_nett = np.expand_dims(np.array(y_nett), axis=1)

#Initialise plot
fig = plt.figure(figsize=(15, 9))
ax = plt.subplot(111)


y_store = []
# Plot individual level_2 industry value
for i, ind in enumerate(industries):
    temp = np.array((df.loc[:, ['value']])[df['level_2']==ind])
    
    differences = np.convolve(temp[:, 0], [1, -1], mode='valid')
    y_ind=np.divide(differences, temp[0:temp.shape[0]-1, 0])*100
    y_store.append(y_ind)


y_plot = np.array(y_store)
y_plot = np.clip(y_plot, -50, 50)

print('industry shape {}, x year shape {}, dat shape {}'.format(industries.shape[0], x_year[:, 0].shape[0], y_plot.shape))

im, cbar = plot.heatmap(y_plot, industries, x_year[1:, 0], ax=ax,
                   cmap="RdBu", cbarlabel="Growth in %/year", vmin=vmin_, vmax=vmax_)
#texts = plot.annotate_heatmap(im, valfmt="{x:.1f} t")
fig.tight_layout()


"""
#Plot original data
plot.plot_nice_lineplot(x_year, y_nett, title='Yearly manufacturing output, millions/SGD', ax=ax, line_label='Original')    

#----- LINEAR MODEL FITTING -----#

#Plot unregularised linear fit
lr = LinearRegression()
lr.fit(x_year, y_nett)
y_fitted = lr.predict(x_year)
plot.plot_nice_lineplot(x_year, y_fitted, ax=ax, col_index=6, line_label='Linear Fit', linestyle='--')

#----- SECOND ORDER MODEL FITTING -----#

#Plot unregularised polynomial fit
poly_lr = LinearRegression()
x_norm = x_year - x_year[0]
x_poly = np.concatenate((x_norm, x_norm**2), axis = 1)
poly_lr.fit(x_poly, y_nett)
y_fitted = poly_lr.predict(x_poly)   
plot.plot_nice_lineplot(x_year, y_fitted, ax=ax, col_index=14, line_label='Square Fit', linestyle='--')


"""

#Show graph
plt.show()