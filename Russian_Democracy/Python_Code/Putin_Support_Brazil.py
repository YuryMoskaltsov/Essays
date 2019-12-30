#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:02:37 2019

@author: yurymoskaltsov
"""

import numpy as np
import math
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.linear_model import *
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.metrics import r2_score
from scipy.interpolate import *


class Data():
    
    '''This is a parent class to all subclasses of data.
    This class contains functions that are general to all datasets
    (i.e. normalization of y values, graphing the results, etc.). '''
    
    def normalizeY(self):
        
        max_val = max(self.getY())
        min_val = min(self.getY())
        
        normalized = [ (value - min_val)/(max_val - min_val) for value in self.getY() ]
        
        return normalized
        
    
    def graph_line(self,title, y_label):
        
        
        plt.figure(figsize = (6,5))
        plt.ylabel(y_label, fontsize = 15)
        plt.xlabel("Year", fontsize = 15)
        plt.xticks(rotation = 90)
        plt.grid(axis = "y")
        plt.title(title, fontsize = 25, fontweight =20)

        return plt.plot(self.getX(), self.getY(), marker = 'o')
    
    def graph_scatter(self,title, y_label):
        
        plt.figure(figsize = (6.2,5))
        plt.ylabel(y_label, fontsize = 15)
        plt.xlabel("Year", fontsize = 15)
        plt.xticks(rotation = 90)
        plt.grid(axis = "y")
        plt.title(title, fontsize = 25, fontweight =20)

        return plt.scatter(self.getX(), self.getY())
    
    def graph_bar(self, title, x_name, y_name):
        
        plt.title(title, fontsize = 20)
        plt.xlabel(x_name, fontsize = 15)
        plt.ylabel(y_name, fontsize = 15)
  
        return sns.barplot( x = self.getX(), y = self.getY())
    
    def graph_box(self):
        
        plt.subplot(211)
        sns.boxplot( x = self.getX())
        plt.subplot(212)
        sns.boxplot( x = self.getY(), color = "orange")
  
        return plt.show()
    
    def normalize_graph(self, title, y_label):
        
        plt.figure(figsize = (6.2,5))
        plt.ylabel(y_label, fontsize = 15)
        plt.xlabel("Year", fontsize = 15)
        plt.title(title, fontsize = 25, fontweight =20)
        
        return plt.plot(self.getX(), self.normalizeY(), marker = 'o')
    
    def r_change(self):
        
        ''' This function outputs the list of rates of change for a given
        y in a dataset'''
        
        r_change = []
#        r_change.append(0)
        
        for i in range(1,len(self.getY())):
            
           r_change.append((self.getY()[i] - self.getY()[i-1])/self.getY()[i-1])
        
        r_change = [0] +r_change
        
        return r_change
    
    def graph_r_change(self,title, y_label):
        
        plt.figure(figsize = (6.2,5))
        plt.ylabel(y_label, fontsize = 15)
        plt.xlabel("Year", fontsize = 15)
        plt.xticks(rotation = 90)
        plt.grid(axis = "y")
        plt.title(title, fontsize = 25, fontweight =20)

        return plt.plot(self.getX(), self.r_change(), marker = 'o')
    
    def graph_compare(self, title, x_name, x_val, var_name, var_values):
        
        ''' Function takes n number of variables and plots them on 1 graph. 
        Assumes that all the varaibles have the same x_value and the size of the 
        lists are the same.
        
        title: graph title (string)
        x_name: name of the values along x-axis (string)
        x_val: values for x (list of floats/ints)
        var_name: names of the dependent varaibles (list of strings)
        var_val: values for dependent varaibles(list of lists of number)'''
        
        
        plt.figure(figsize =(8,7))
        
        for i in range(len(var_name)):
                
            plt.plot(x_val, var_values[i], label = var_name[i], marker = "o")
            plt.xlabel(x_name)
        
        plt.legend()
        plt.title(title)
        
#    def graph_mean_bar(self, title, x_label, y_label, dataset1,dataset2, add_column, group_column):
#        
#        dataset2[add_column] = dataset1.getY()
#        grouped_dataset = dataset.groupby(add_column)[group_column].mean().reset_index()
#        plt.xlabel(x_label)
#        plt.ylabel(y_label)
#        plt.title(title, fontsize = 18)
#        plt.bar(gini_v_putin_grouped[group_column], gini_v_putin_grouped[add_column])
            
def normalizeAny(data):
            
        max_val = max(data)
        min_val = min(data)
        
        normalized = [ (value - min_val)/(max_val - min_val) for value in data]
        
        return normalized
    
    
    
class WorldBankData(Data):
    
    ''' This class is specific to the data format from the excel
    files downloaded from the WorldBank website. It allows to search
    for the information of the specific countries for the specific 
    period.'''
    
    def __init__(self,name, country_name,dataframe, period_start, period_end):
        
        self.name = name
        self.country_name = country_name
        self.df = dataframe
        self.period_start = period_start
        self.period_end = period_end
            
    
    def clean_data(self):
        
        
        
        df_country = self.df[self.df["Country Name"] == self.country_name] 
#    df_country = df_country.fillna(0)
    
        if len(df_country) == 0:
            
            raise ValueError("No such country in the list")
    
        df_country_edited = df_country.drop(["Country Name", "Country Code","Indicator Name", "Indicator Code"], axis = 1)
        
        years = df_country_edited.columns
        values = np.array(df_country_edited)
        years_new = []
        values_new = []
        
        for index in range(len(years)):
            
            if  self.period_start < int(years[0]) :
                
                raise IndexError("The data is only available from " + years[0])
            
            if  self.period_end > int(years[-1]):
                
                raise IndexError("The data is only available till " + years[-1])
            
            else:
                
                if int(years[index]) >= self.period_start and int(years[index]) <= self.period_end:
                    
                    years_new.append(years[index])
                    values_new.append(round(values[0][index],0))         
                                 
        years_new = np.array(years_new)
        years_new = years_new.reshape(-1,1)
        
        values_new = np.array(values_new)
        values_new = values_new.reshape(-1,1)
    
               
        df_new = pd.DataFrame(years_new, columns = ['Year'] )
        df_new['values'] = values_new
        
        return df_new
    
    def mean_line(self):
        
        ''' Creates a list with the number of repeated mean values 
        that correspond to the number of years. This is done to plot
        a mean line in the plot function'''
        
        mean = self.clean_data()['values'].mean()
        mean_line = []
        
        for i in self.clean_data()['values']:
            
            mean_line.append(round(mean,2))
        
        return mean_line
    
    def median_line(self):
        
        median = self.clean_data()['values'].median()
        median_line = []
        
        for i in self.clean_data()['values']:
            
            median_line.append(round(median,2))
        
        return median_line
    
    def getX(self):
        
        return list(self.clean_data()['Year'])        
    
    def getY(self):
        
        return list(self.clean_data()['values'])
        
        
df_gini = pd.read_excel("Gini_coefficient_world.xls")
brazil_gini = WorldBankData("Gini Index","Brazil", df_gini, 2003, 2018)
print(brazil_gini.clean_data())



df_gdp = pd.read_excel("GDP_Per_Capita.xls")

brazil_gdp = WorldBankData("GDP Per Capita","Brazil", df_gdp, 2003, 2013)
#russia_gdp.graph_line("Russia GDP", "GDP (USD Thousands)" )
print(brazil_gdp.clean_data())

df_poverty = pd.read_excel("Poverty_World_Bank.xls")
brazil_poverty = WorldBankData("Poverty","Brazil", df_poverty, 2003, 2013)
print(brazil_poverty.clean_data())




class TimeIndexData(Data):
    
    ''' This class takes the dataset that has been timedate indexed 
    and resamples the data to be a yearly data and the values 
    corresponding to each year are the respective average of all the dates 
    for the corresponding year. '''
    
    
    def __init__(self, dataset, x_name, y_name):
        
        ''' Class takes three parameters
        
        dataset: timeindexed dataset
        x_name: name of the x column
        y_name: name of the y column'''
        
        self.dataset = dataset
        self.x_name = x_name
        self.y_name = y_name
    
    def clean_data(self):
        
        ''' So for this dataset I had daily prices of oil from 2000
    clean data outputs the data where instead of the daily prices 
    I have the average price for a year'''
    
        self.dataset.rename(columns ={"Closing Value": "Price", "Date": "Year"}, inplace = True)
        self.dataset = self.dataset.resample("Y").mean()
        self.dataset["Year"] = self.dataset.index.year
        
        return self.dataset
    
    def getX(self):
        
        return list(self.clean_data()[self.x_name])
                    
    def getY(self):
        
        return list(self.clean_data()[self.y_name])


df_oil_price = pd.read_excel("Oil_Prices_From_2000.xlsx", parse_dates = ["Date"], index_col = "Date")
russia_oil_price = TimeIndexData(df_oil_price, "Year", "Price")
print(russia_oil_price.clean_data())
#print(russia_oil_price.graph_line("Annual Oil Price", "Oil Price (USD)"))
#print(russia_oil_price.getX())
#print(russia_oil_price.getY())
#print(russia_oil_price.normalizeY())
#print(russia_oil_price.normalize_graph("Annual Oil Price", "Oil Price (USD)"))

class GeneralData(Data):
    
    '''This is a class for general sorted data where we have necessary 
    columns, so it takes three parameters
    
    1) Dataset with the info
    2) Name of x column
    3) Name of Y column
    4) Start period
    5) End period'''
    
    def __init__(self, dataset, x, y, start, end):
    
        self.dataset = dataset
        self.x = x
        self.y = y
        self.start = start
        self.end = end
    
    def clean_data(self):
        
        x = list(range(self.start,self.end+1))
        y = []
        
        if  self.start < int(list(self.dataset[self.x])[0]) :
                
            raise IndexError("The data is only available from " + str(list(self.dataset[self.x])[0]))
            
        if  self.end > int(list(self.dataset[self.x])[-1]):
            
            raise IndexError("The data is only available till " + str(list(self.dataset[self.x])[-1]))
        
        for i in range(len(self.dataset[self.x])):
            
            
            if int(self.dataset[self.x][i]) in range(self.start,self.end+1):
                
                y.append(round(self.dataset[self.y][i],2))
            
        return x,y
    
    def getX(self):
        
        x,y = self.clean_data()
        
        return x
    
    def getY(self):
        
        x,y = self.clean_data()
        
        return y
    
 
        
'''Poverty Rate'''
df_poverty_rate = pd.read_excel("Number_of_poor_people.xls")
poverty_rate = GeneralData(df_poverty_rate, "Year", "Poverty %", 2000,2018)
#print(df_poverty_rate)
#print("Poverty Rate: ", poverty_rate.clean_data())
#print(poverty_rate.getX())
#print(poverty_rate.getY())
#print(poverty_rate.graph_line("Russia Poverty Rate", "% population below poverty line"))
#print(poverty_rate.normalizeY())
#print(poverty_rate.normalize_graph("Russia Poverty Rate", "% population below poverty line"))

'''Freedom House Democracy Score'''
df_russia_dem_score = pd.read_excel("Russia_Democracy_Score.xlsx")
russia_dem_score = GeneralData(df_russia_dem_score, "Year", "Democracy Score", 2003,2013)
print(russia_dem_score.clean_data())
#russia_dem_score.graph_line("FH Democracy Score", "FH Democracy Score")

#plt.savefig("/Users/yurymoskaltsov/Desktop/Test.pdf")


class OutputData(Data):
    
    def __init__(self, dataset, x, y):
    
        self.dataset = dataset
        self.x = x
        self.y = y
    
    def clean_data(self):
        
        ''' This function checks if the years are in the descending 
        order and if they are it reverts it to ascending order'''
        
        self.dataset[self.dataset.columns[0]].apply(lambda x: int(x))
#        print(self.dataset.iloc[0,0]+ self.dataset.iloc[1,0])
        
        if self.dataset.iloc[0,0] > self.dataset.iloc[1,0]:
            
         self.dataset = self.dataset.sort_values(self.dataset.columns[0], ascending = True).reset_index()
         self.dataset.drop([self.dataset.columns[0]], axis =1, inplace = True)
         self.dataset.drop(self.dataset.tail(1).index, inplace = True)
        
        return self.dataset
     
    def getX(self):
        
        return list(self.clean_data()[self.x])
    
    def getY(self):
        
        return list(self.clean_data()[self.y])
     
#    def least_squares(self):
#        
#        ''' This function outputs minimised error between the actual
#        graph points and the modeled ones'''
#        
#        x = 3
#        value = np.pi
#        sum_squares = 0
#        smallest_error = 3
#        best_value = 0
#        step = (12*np.pi)/(19*2)
##        print(len(self.normalize_x()))
#        
#        for i in range(300):
#            
#            
#            for i in range(19):
#               
#                sum_squares +=  (self.normalizeY()[i] - np.sin(value))**2
#                value+= step
#            
#            error = math.sqrt(sum_squares)
#            print(error)
#            if error < smallest_error:
#                
#                smallest_error = error
#                best_value = value
#            
#            value = np.pi/(x+step)
#            
#        return smallest_error, best_value
       
df_putin_support = pd.read_excel("Putin_Support.xlsx")
#df_putin_support.sort_values("Year", ascending = True)
#print(df_putin_support.iloc[:,:])
x_put_sup = df_putin_support.Year
y_put_sup = df_putin_support.Support
#print(df_putin_support)
#print(putin_support.clean_data())
#print(y_put_sup)


putin_support = OutputData(df_putin_support[5:-3], "Year", "Support")
#print(putin_support.normalize_x())
#print(putin_support.normalizeY())
#print(putin_support.getY())
#print(putin_support.normalize_graph("Annual Putin Support Poles", "% of support"))
#print(putin_support.least_squares())
#print(putin_support.clean_data())
#print(putin_support.graph_scatter("Annual Putin Support Poles", "% of support"))
#print(putin_support.graph_line("Annual Putin Support Poles", "% of support"))

def sin_graph():
    
    ''' Here I am trying to approximate the function for the 
    putin_support data in order to use it as a prediction model.
    We know that the graph is cyclical, so we can use the sin(x)
    fucntion. There are 2.5 cycles. So we need to figure out the 
    total distance and the distance for every cycle and then divide
    each cycle distance by the amount of points this cycle has which
    should correspond to number of years'''
    
    y = []
    x = range(2000,2019)
#    first_cycle = 5
#    second_cycle = 8
#    third_cycle = 5
#    first_cycle_int_norm = 1/abs(max(y[:first_cycle+1]) - min(y[:first_cycle+1]))
#    first_cycle = 5
#    second_cycle = 8
#    print('First cycle int: ', first_cycle_int_norm)
    
    y1 = []
    y2 = []
    y3 = []
    
    value = -0.1
    
    for value in range(2000, 2006):
        
        
        y1.append(((np.sin(value/3-1.5))/3+np.cos(value+3))/3+value/5000)
    
    value = 0.35
    
    for value in range(2006, 2014):
        
        
        y2.append(((np.sin(value/3-1.5))/3+np.cos(value+3))/3+value/5000)
    
    value = 0.6
    
    for i in range(2014, 2019):
        
        
        y3.append(((np.sin(value/3-1.5))/3+np.cos(value+3))/3+value/5000)
        
    
#    y.append(value)
    
    
        
        
    
#    del y[1]
    return y1,y2,y3
    
#    plt.figure(figsize = (10,8))
#    plt.title("Sin Graph", fontsize = 25)
#    plt.plot(x,y)
    
    

#def compare_graphs():
#    
#    x = range(2000,2019)
#    y = sin_graph()
#    z = putin_support.normalizeY()
#    plt.plot(x,y, label = "Sin graph", color = "orange", marker = "o")
##    plt.plot(x_sin2,y2, label = "Sin graph", color = "red")
##    plt.plot(x_sin3,y3, label = "Sin graph", color = "aqua")
#    plt.scatter(x,z, label = "Normalized Values Graph")
#    plt.legend(loc = "upper left")
#    plt.title("Graph Compare", fontsize = 20)
#    
#    return plt.show()

def compare_graphs():
    
    x_sin1 = range(2000,2006)
    x_sin2 = range(2006,2014)
    x_sin3 = range(2014,2019)
    x_actual= range(2000,2019)
    y1,y2,y3 = sin_graph()
    z = putin_support.normalizeY()
    plt.plot(x_sin1,y1, label = "Sin graph", color = "orange")
    plt.plot(x_sin2,y2, label = "Sin graph", color = "red")
    plt.plot(x_sin3,y3, label = "Sin graph", color = "aqua")
    plt.scatter(x_actual,z, label = "Normalized Values Graph")
    plt.legend(loc = "upper left")
    plt.title("Graph Compare", fontsize = 20)
    
    return plt.show()

#print(sin_graph())
#print(len(sin_graph()))
#print(compare_graphs())

#plt.clf()
#plt.plot(x, y)
#plt.plot(x, z)
#plt.show()
    
class Regression(Data):
    
    ''' This class contains different function that calculate necessary
    values for regression analysis(i.e. standard deviation, covariance,
    linear/polynomial regression, etc.)'''
    
    
    def __init__(self,x_name, y_name, x, y):
        
        self.x_name = x_name
        self.y_name = y_name
        self.x = x
        self.y = y
    
    
    def getX(self):
        
        return self.x
    
    def getY(self):
    
        return self.y
    
    
    def linear_regr(self):
        
        x_regr= np.array(self.getX()).reshape(-1,1)
        y_regr = np.array(self.getY()).reshape(-1,1)
        X = []
        
        line_fitter = LinearRegression()
        line_fitter.fit(x_regr, y_regr)
        
        return line_fitter.predict(x_regr)
    
    
    def polynomial_regr(self, power):
        
        
        x_regr = self.getX()
        y_regr = self.getY()
        
        
        p = np.polyfit(x_regr,y_regr, power)
#        print
#        plt.plot(x_regr, np.polyval(p4,x_regr), "r", linewidth = 4)
#        plt.plot(x_regr, np.polyval(p6,x_regr), "g", linewidth = 4)
#        plt.scatter(x_regr,y_regr)
#        np.polyval(p6,x_regr)
        
        return np.polyval(p,x_regr)
        
    def error_line(self,y_predicted):
        
        ''' This function outputs minimised error between the actual
        graph points and the modeled ones'''
        
        error = 0
        
        
        for i in range(len(y_predicted)):
            
            error +=  (self.getY()[i]- y_predicted[i])**2
            
        
       
        
#        print("Total error: ", round(error,2))
        return error
    
    def error_mean(self):
        
        variance = 0
        mean = np.mean(self.getY())
        
        for i in range(len(self.getY())):
            
#            print("Sum of squares: ", sum_squares)
            variance +=  (self.getY()[i]-mean)**2
            
        
        return variance
        
        
    
    def standard_deviationX(self):
        
        variance = 0
        mean = np.mean(self.getX())
        
        for i in range(len(self.getX())):
            
#            print("Sum of squares: ", sum_squares)
            variance +=  (self.getX()[i]-mean)**2
            
        
        std = math.sqrt(variance/(len(self.y)-1))
        
        return std
    
    
    def covariance(self):
        
        sum_xy = 0
        sum_x = sum(self.getX())
        sum_y = sum(self.getY())
        
        for i in range(len(self.getX())):
            
            sum_xy += self.getX()[i]*self.getY()[i]
        
        covariance = len(self.getX())*sum_xy - (sum_x*sum_y)
        
        return covariance
    
    
    def corr_coef(self):
        
        return np.corrcoef(self.getX(), self.getY())
    
    
    def r_squared(self,y_predicted):
        
        return 1 - self.error_line(y_predicted)/(self.error_mean())
    
    def r_squared_python(self,y_predicted):
        
        return r2_score(self.y, y_predicted)
        
        
    
    def corr_graph_line(self, y_predicted):
        
        
#        plt.clf()
        plt.figure(figsize = (6.2,5))
        plt.xlabel(self.x_name, fontsize = 10)
        plt.ylabel(self.y_name, fontsize = 10)
        plt.title("Correlation "+ self.x_name + " " + self.y_name, fontsize = 15)
        plt.scatter(self.getX(), self.getY())
        plt.plot(self.getX(), y_predicted, color = "Orange", linewidth = 3.5)
    
    def corr_graph_scatter(self, y_predicted):
        
        
#        plt.clf()
        plt.figure(figsize = (6.2,5))
        plt.xlabel(self.x_name, fontsize = 15)
        plt.ylabel(self.y_name, fontsize = 15)
        plt.title("Correlation "+ self.x_name + " " + self.y_name, fontsize = 20)
        plt.scatter(self.getX(), self.getY())
        plt.plot(self.getX(), y_predicted, color = "Orange", linewidth = 3.5)
        
        
        return plt.show()
    
#    def graph_bar(self, x_graph,y_graph):
#        
#        print(self.x_name)
#        plt.figure(figsize = (6.2,5))
#        sns.barplot(x = x_graph , y = y_graph)
#        plt.xlabel(self.x_name, fontsize = 15)
#        plt.ylabel(self.y_name, fontsize = 15)
#        plt.title(self.x_name + " " + "vs " + self.y_name, fontsize = 25)
##        plt.scatter(self.getX(), self.getY())
#    
#        return plt.show()
  







