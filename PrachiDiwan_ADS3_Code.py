# -*- coding: utf-8 -*-
"""
Created on Thu Jan 5 17:03:36 2023

@author: prach
"""


"""
CLUSTERING PART STARTS
"""

# import necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import sklearn.metrics as skmet


# function to read file
def readFile(x):
    '''
        This function is used to read csv file in original form and then 
        filling 'nan' values by '0' to avoid disturbance in data visualization
        and final results.

        Parameters
        ----------
        x : csv filename
    
        Returns
        -------
        pop_growth : variable for storing csv file

    '''
    pop_growth = pd.read_csv("Population_growth.csv");
    pop_growth = pd.read_csv(x)
    pop_growth = pop_growth.fillna(0.0)
    return pop_growth
 
# calling readFile function to display dataframe 
pop_growth = readFile("Population_growth.csv")

print("\nPopulation Growth: \n", pop_growth)


# dropping particular columns which are not required to clean data
pop_growth = pop_growth.drop(['Country Code', 'Indicator Name', 'Indicator Code', '1960'], axis=1)

print("\nPopulation Growth after dropping columns: \n", pop_growth)


# transpose dataframe
pop_growth = pd.DataFrame.transpose(pop_growth)

print("\nTransposed Population Growth: \n",pop_growth)


# populating header with header information
header1 = pop_growth.iloc[0].values.tolist()

pop_growth.columns = header1

print("\nPopulation Growth Header: \n", pop_growth)


# remove first two rows from dataframe
pop_growth = pop_growth.iloc[2:]

print("\nPopulation Growth after selecting particular rows: \n", pop_growth)


# creating a dataframe for two columns to store original values
pop_ex = pop_growth[["India","Greece"]].copy()


# extracting maximum and minmum value from new dataframe
max_val = pop_ex.max()

min_val = pop_ex.min()

pop_ex = (pop_ex - min_val) / (max_val - min_val) # operation of min and max

print("\nMin and Max operation on Population Growth: \n", pop_ex)


# set up clusterer and number of clusters
ncluster = 5

kmeans = cluster.KMeans(n_clusters=ncluster)


# fitting the data where the results are stored in kmeans object
kmeans.fit(pop_ex)

labels = kmeans.labels_ # labels is number of associated clusters


# extracting estimated cluster centres
cen = kmeans.cluster_centers_

print("\nCluster Centres: \n", cen)


# calculate the silhoutte score
print("\nSilhoutte Score: \n",skmet.silhouette_score(pop_ex, labels))


# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))

col = ["tab:red", "tab:pink", "tab:green", "tab:blue", "tab:brown", \
       "tab:cyan", "tab:orange", "tab:black", "tab:olive", "tab:gray"]

    
# loop over the different labels    
for l in range(ncluster): 
    plt.plot(pop_ex[labels==l]["India"], pop_ex[labels==l]["Greece"], 
             marker="o", markersize=3, color=col[l])    

    
# display cluster centres
for ic in range(ncluster):
    xc, yc = cen[ic,:]  
    plt.plot(xc, yc, "dk", markersize=10)

plt.xlabel("India")

plt.ylabel("Greece")

plt.show()    


print("\nCentres: \n", cen)


df_cen = pd.DataFrame(cen, columns=["India", "Greece"])

print(df_cen)

df_cen = df_cen * (max_val - min_val) + max_val

pop_ex = pop_ex * (max_val - min_val) + max_val
# print(df_ex.min(), df_ex.max())

print("\nDataframe Centre: \n", df_cen)


# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))

col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", \
"tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

for l in range(ncluster): # loop over the different labels
    plt.plot(pop_ex[labels==l]["India"], pop_ex[labels==l]["Greece"], "o", markersize=3, color=col[l])
    

# show cluster centres
plt.plot(df_cen["India"], df_cen["Greece"], "dk", markersize=10)

plt.xlabel("India")

plt.ylabel("Greece")

plt.title("Population Growth(%)")

plt.show()

print("\nCentres: \n", cen)    




# In[ ]:
    
"""
CURVE FIT PART STARTS
"""


# import necessary modules
import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
#import errors as err


# function to read file
def readFile(y):
    '''
        This function is used to read csv file in original form and then 
        filling 'nan' values by '0' to avoid disturbance in data visualization
        and final results.

        Parameters
        ----------
        x : csv filename
    
        Returns
        -------
        gdp_growth : variable for storing csv file

    '''
    gdp_growth = pd.read_csv("GDP_Growth.csv");
    gdp_growth = pd.read_csv(y)
    gdp_growth = gdp_growth.fillna(0.0)
    return gdp_growth


# calling readFile function to display dataframe 
gdp_growth = readFile("GDP_Growth.csv")

print("\nGDP Growth: \n", gdp_growth)


# converting to dataframe
gdp_growth = pd.DataFrame(gdp_growth)


# transpose dataframe
gdp_growth = gdp_growth.transpose()

print("\nTransposed GDP Growth: \n", gdp_growth)


# populating header with header information
header2 = gdp_growth.iloc[0].values.tolist()

gdp_growth.columns = header2

print("\nGDP Growth Header: \n", gdp_growth)


# select particular column
gdp_growth = gdp_growth["Bahamas, The"]

print("\nGDP growth after selecting particular column: \n", gdp_growth)


# rename column
gdp_growth.columns = ["GDP"]

print("\nRenamed GDP Growth: \n", gdp_growth)


# extracting particular rows
gdp_growth = gdp_growth.iloc[5:]

gdp_growth = gdp_growth.iloc[:-1]

print("\nGDP after selecting particular rows: \n", gdp_growth)


# resetn index of dataframe
gdp_growth = gdp_growth.reset_index()

print("\nGDP Growth reset index: \n", gdp_growth)


# rename columns
gdp_growth = gdp_growth.rename(columns={"index": "Year", "Bahamas, The": "GDP"} )

print("\nGDP Growth after renamed columns: \n", gdp_growth)

print(gdp_growth.columns)


# plot line graph
gdp_growth.plot("Year", "GDP", label="GDP")

plt.legend()

plt.title("GDP Growth")

plt.show()


# curve fit with exponential function
def exponential(s, q0, h):
    '''
        Calculates exponential function with scale factor n0 and growth rate g.
    '''
    s = s - 1960.0
    x = q0 * np.exp(h*s)
    return x


# performing best fit in curve fit
print(type(gdp_growth["Year"].iloc[1]))

gdp_growth["Year"] = pd.to_numeric(gdp_growth["Year"])

print("\nGDP Growth Type: \n", type(gdp_growth["Year"].iloc[1]))

param, covar = opt.curve_fit(exponential, gdp_growth["Year"], gdp_growth["GDP"],
p0=(4.978423, 0.03))


# plotting best fit
gdp_growth["fit"] = exponential(gdp_growth["Year"], *param)

gdp_growth.plot("Year", ["GDP", "fit"], label=["GDP", "Fit"])

plt.xlabel("Year")

plt.ylabel("GDP")

plt.legend()

plt.title("GDP Growth")

plt.show()


# predict fit for future years
year = np.arange(1960, 2031)

print("\nForecast Years: \n", year)

forecast = exponential(year, *param)

plt.figure()

plt.plot(gdp_growth["Year"], gdp_growth["GDP"], label="GDP")

plt.plot(year, forecast, label="Forecast")

plt.xlabel("Year")

plt.ylabel("GDP")

plt.title("GDP Growth")

plt.legend()

plt.show()


# err_ranges function
def err_ranges(x, exponential, param, sigma):
    '''
        Calculates the upper and lower limits for the function, parameters and
        sigmas for single value or array x. Functions values are calculated for 
        all combinations of +/- sigma and the minimum and maximum is determined.
        Can be used for all number of parameters and sigmas >=1.
    
        This routine can be used in assignment programs.
    '''
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = exponential(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = exponential(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        print("\nLower: \n", lower)
        print("\nUpper: \n", upper)        
    return lower, upper




