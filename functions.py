#This will contain our functions that we will call to the main project
print("this is test")
#Importing the nesessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Function to read the datasets
def dataframe(data):
    df= pd.read_csv(data)
    return df
#Function to check the properies of the dataset.
def properties(data):
    shape=data.shape
    duplicates=data.duplicated().sum()
    info=data.dtypes
    return shape,duplicates,info
# Function to filter out african countries in all our datasets
def african_countries(data1,data2,column1,column2):
    codes=list(data2[column2])
    df=data1[data1[column1].isin(codes)]
    return df
# Function to check for percentage of missing values
def missing_percentage(data):
    p_miss=((data.isnull().sum() / len(data))*100)
    missing_value_df=pd.DataFrame({'column_name': data.columns,'percent_missing': p_miss})
    return missing_value_df
# Function to reset index after cleaning
def reset_index(data):
    df=data.reset_index().drop(columns=['index'])
    return df
# Function to export clean dataframes
def export(data,name):
    dataset=data.to_csv(name)
#Importing libraries for the Q-Qplot
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
# Function for the Q-Q plot
def normality(data,column):
  qq=qqplot(data[column], line='s')
  qq=pyplot.show()
  return qq
# Function for the distplot
def distribution(data,column):
  plt.figure(figsize=(13,4))
  sns.distplot(data[column],kde=True)
  plt.xlabel(column)
  plt.ylabel(format(column)+'distribution')
  plt.title('The distribution of '+format(column))
# Function for performing a boxplot
def boxplot(data,column):
    plt.figure(figsize=(13,4))
    sns.boxplot(data[column])
    plt.xlabel(column)
    plt.ylabel(format(column)+'values')
    plt.title('A box plot of  '+format(column))
# Function for plotting a scattterplot
def scatterplot(data,column1,column2,title):
    plt.scatter(data[column1],data[column2])
    plt.title(title)
    plt.xlabel(column1)
    plt.ylabel(column2)
    fig = plt.figure(figsize = (15, 7))
    plt.show()
# A function to tabulate the test statistic of two sample means
def Z_test(X1, X2, sigma1, sigma2, N1, N2):
    from numpy import sqrt, abs, round
    from scipy.stats import norm
    sigma = sqrt(sigma1**2/N1 + sigma2**2/N2)
    z = (X1 - X2)/sigma
    pval = 2*(1 - norm.cdf(abs(z)))
    return z, pval