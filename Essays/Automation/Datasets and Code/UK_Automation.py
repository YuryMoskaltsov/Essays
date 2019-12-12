#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 21:40:56 2019

@author: yurymoskaltsov
"""

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
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split 
from scipy.interpolate import *
from scipy.optimize import curve_fit


class Data():
    
    '''This is a parent class to all subclasses of data.
    This class contains functions that are general to all datasets
    (i.e. normalization of y values, graphing the results, etc.). '''
    
    def normalizeX(self):
        
        max_val = max(self.getX())
        min_val = min(self.getX())
        
        normalized = [ (value - min_val)/(max_val - min_val) for value in self.getX() ]
        
        return normalized
      
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

        return plt.plot(self.getX(), self.getY(), marker = "o")
    
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
        
    def graph_table(self, rows, columns, values):
        
        plt.clf()
        table = plt.table(rowLabels = rows, colLabels = columns, \
                    cellText = values, cellLoc = 'center',\
                    loc = 'center', colWidths = [0.4]*len(columns))
        table.scale(2,8)
#        table.auto_set_font_size(False)
#        table.set_font_size(20)
        plt.axis('off')
        
        plt.show()
            
  
        return plt.show()
    
    def normalize_graph(self, title, y_label):
        
        plt.figure(figsize = (6.2,5))
        plt.ylabel(y_label, fontsize = 15)
        plt.xlabel("Year", fontsize = 15)
        plt.title(title, fontsize = 25, fontweight =20)
        
        return plt.plot(self.getX(), self.normalizeY(), marker = "o")
    
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
        plt.title(title, fontsize = 25, fontweight =20, marker = "o")

        return plt.plot(self.getX(), self.r_change())
    
    def graph_compare(self, title, x_name, x_val, var_name, var_values):
        
        ''' Function takes n number of variables and plots them on 1 graph. 
        Assumes that all the varaibles have the same x_value and the size of the 
        lists are the same.
        
        title: graph title (string)
        x_name: name of the values along x-axis (string)
        x_val: values for x (list of floats/ints)
        var_name: names of the dependent varaibles (list of strings)
        var_val: values for dependent varaibles(list of lists of number)'''
        
        
#        plt.figure(figsize =(8,7))
        
        for i in range(len(var_name)):
                
            plt.plot(x_val, var_values[i], label = var_name[i])
            plt.xlabel(x_name)
        
        plt.legend()
        plt.title(title)
        
            
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
    

df_gdp = pd.read_excel("GDP_Per_Capita.xls")
uk_gdp = WorldBankData("GDP Per Capita","United Kingdom", df_gdp, 2003, 2016)




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
'''GVA'''
df_gva = pd.read_excel("GVA(Cleaned).xlsx")
gva_retail = GeneralData(df_gva, "Year", "Retail", 2001, 2017) 
gva_admin = GeneralData(df_gva, "Year", "Administrative", 2001, 2017)
gva_science = GeneralData(df_gva, "Year", "Scientific research and development", 2001, 2017)
gva_manuf = GeneralData(df_gva, "Year", "Manufacturing", 2001, 2017)

#'''GVA By Sector Nominal'''
#gva_retail.graph_compare("GVA by Sector", "Year", gva_retail.getX(), ["Retail", "Administrative", "Scientific Research", "Finance" ], \
#[gva_retail.getY(), gva_admin.getY(), gva_science.getY(), gva_finance.getY()])


#'''GVA By Sector Normalised'''
#gva_retail.graph_compare("GVA by Sector", "Year", gva_retail.getX(), ["Retail", "Administrative", "Scientific Research", "Manufacturing"], \
#[gva_retail.normalizeY(), gva_admin.normalizeY(), gva_science.normalizeY(), gva_manuf.normalizeY() ])

'''Employment'''
df_employment = pd.read_excel("Employment_By_Sector.xls")
df_employment = df_employment.groupby("Year").mean().reset_index()
empl_retail =  GeneralData(df_employment, "Year", "Retail", 2001, 2017)
empl_admin = GeneralData(df_employment, "Year", "Administrative", 2001, 2017)
empl_science = GeneralData(df_employment, "Year", "Science", 2001, 2017)
empl_manuf = GeneralData(df_employment, "Year", "Manufacturing", 2001, 2017)


'''Employment By Sector Nominal'''
#empl_retail.graph_compare("Employment by Sector", "Year", empl_retail.getX(), ["Retail", "Administrative", "Scientific Research", "Finance" ], \
#[empl_retail.getY(), empl_admin.getY(), empl_science.getY(), empl_finance.getY()])
#
'''Employment By Sector Normalised'''
#empl_retail.graph_compare("Employment by Sector", "Year", empl_retail.getX(), ["Retail", "Administrative", "Scientific Research", "Manufacturing" ], \
#[empl_retail.normalizeY(), empl_admin.normalizeY(), empl_science.normalizeY(), empl_manuf.normalizeY()])

'''Research and Development'''
df_rd = pd.read_excel("RD_Expenditure(Main).xlsx")
rd_retail = GeneralData(df_rd, "Year", "Retail", 2001, 2017)
rd_admin = GeneralData(df_rd, "Year", "Administrative", 2001, 2017)
rd_science = GeneralData(df_rd, "Year", "Computer Programming", 2001, 2017)
rd_manuf = GeneralData(df_rd, "Year", "Manufacturing", 2001, 2017)

#'''RD By Sector Nominal'''
#rd_retail.graph_compare("R&D by Sector", "Year", rd_retail.getX(), ["Retail", "Administrative", "Scientific Research", "Finance" ], \
#[rd_retail.getY(), rd_admin.getY(), rd_science.getY(), rd_finance.getY()])
#
#'''RD By Sector Normalised'''
#rd_retail.graph_compare("R&D by Sector", "Year", rd_retail.getX(), ["Retail", "Administrative", "Scientific Research", "Manufacturing"], \
#[rd_retail.normalizeY(), rd_admin.normalizeY(), rd_science.normalizeY(), rd_manuf.normalizeY()])



class TimeIndexData(Data):
    
    ''' This class takes the dataset that has been timedate indexed 
    and resamples the data to be a yearly data and the values 
    corresponding to each year are the respective average of all the dates 
    for the corresponding year. '''
    
    
    def __init__(self, dataset, x_name, y_name, start, end):
        
        ''' Class takes three parameters
        
        dataset: timeindexed dataset
        x_name: name of the x column
        y_name: name of the y column'''
        
        self.dataset = dataset
        self.x_name = x_name
        self.y_name = y_name
        self.years = list(range(start,end+1))
    
    def clean_data(self):
        
        '''This dataset takes daily prices and
        clean data outputs the average price for a year
        instead of the daily prices '''
    
        self.dataset.rename(columns ={"Closing Value": "Price", "Date": "Year"}, inplace = True)
        self.dataset = self.dataset.resample("Y").mean()
        self.dataset["Year"] = self.dataset.index.year
        
        return self.dataset[self.dataset[self.x_name].isin(self.years)]
    
    def getX(self):
        
        return list(self.clean_data()[self.x_name])
                    
    def getY(self):
        
        return list(self.clean_data()[self.y_name])

df_ftse = pd.read_excel("FTSE100.xlsx", parse_dates = ["Date"], index_col = "Date")
#print(df_ftse.head(-1))
ftse = TimeIndexData(df_ftse, "Year", "Price", 2001,2017)
#ftse.graph_line("FTSE", "Annual FTSE Values")
#print('-'*10)



class SimpleRegression(Data):
    
    ''' This class contains different function that calculate necessary
    values for regression analysis(i.e. standard deviation, covariance,
    polynomial regression, etc.)'''
    
    
    def __init__(self,x_name, y_name, x, y):
        
        self.x_name = x_name
        self.y_name = y_name
        self.x = x
        self.y = y
    
    
    def getX(self):
        
        return self.x
    
    def getY(self):
    
        return self.y
    
    
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
    
    def expo_regr(self):
        
        def func(x, a, b, c):
            
            return a* np.exp(-b * x) + c
        
        x_regr = self.getX()
        y_regr = self.getY()
        
        
        popt,pcov = curve_fit(func, x_regr,y_regr)
        y_predicted = []
        for i in x_regr:
            
            y_predicted.append(func(i, *popt))
        
        
#        plt.scatter(self.normalizeX(),self.normalizeY())
#        plt.plot(x_regr, y_predicted , 'g--')
        
        return y_predicted
    
    def corr_coef(self):
        
        return np.corrcoef(self.getX(), self.getY())
    
    
    def r_squared(self,y_predicted):
        
        return r2_score(self.y, y_predicted)
        
        
    
    def corr_graph_line(self, y_predicted):
        
        
#        plt.clf()
#        plt.figure(figsize = (6.2,5))
        plt.xlabel(self.x_name, fontsize = 11)
        plt.ylabel(self.y_name, fontsize = 11)
        plt.title(self.x_name, fontsize = 13, fontweight = "bold")
        plt.scatter(self.getX(), self.getY())
        plt.plot(self.getX(), y_predicted, color = "Orange", linewidth = 3.5)
        plt.text(min(self.getX()),max(self.getY()),'R- Squared ' + str(round(self.r_squared(y_predicted),5)),ha = "left", fontsize = "x-large")        
        plt.show()
    
    def corr_graph_scatter(self, y_predicted):
        
        
#        plt.clf()
        plt.figure(figsize = (6.2,5))
        plt.xlabel(self.x_name, fontsize = 15)
        plt.ylabel(self.y_name, fontsize = 15)
        plt.title("Correlation "+ self.x_name + " " + self.y_name, fontsize = 20)
        plt.scatter(self.getX(), self.getY())
        plt.plot(self.getX(), y_predicted, color = "Orange", linewidth = 3.5)
        
        


class MultiRegression(SimpleRegression):
    
    def __init__(self, dataset, x, y, start, end):
        
        '''x is the list of column names to be independent varaibles
        y is the column name for dependant variable''' 
        
        
        self.x = x
        self.y = y
        self.dataset = dataset
        self.start = start
        self.end = end
        self.years = list(range(self.start,self.end+1))
    
    def x_test_data(self):
        
        test_data = self.dataset[self.dataset["Year"].isin([self.end+1])]
        test_data = test_data.drop(columns = "Year")
        return test_data[self.x]
    
    def y_test_data(self):
        
        test_data = self.dataset[self.dataset["Year"].isin([self.end+1])]
        test_data = test_data.drop(columns = "Year")
        return test_data[self.y]
    
    def clean_data(self):
        
        return self.dataset[self.dataset["Year"].isin(self.years)]
        
        
    def corr_table(self):
        
        return self.clean_data()[self.x+self.y].corr()
    
    def mult_lin_regr(self):
        
        x_regr = self.clean_data()[self.x]
        y_regr = self.clean_data()[self.y]
        
#        X_train, X_test, y_train, y_test = train_test_split(x_regr, y_regr, test_size=0.2, random_state= None)
        
        X_train = x_regr[:int(len(x_regr)*0.8)]
        y_train = y_regr[:int(len(y_regr)*0.8)]
#        X_test = x_regr[int(len(x_regr)*0.8):]
#        y_test = y_regr[int(len(y_regr)*0.8):]
#        X_test = self.test_data
        
        model = linear_model.LinearRegression()
#        model.fit(X_train, y_train)
        model.fit(x_regr, y_regr)
        y_predict = model.predict(self.x_test_data())
#        
#        result = pd.DataFrame({"Actual": list(np.array(y_test).flatten()), "Predicted": list(y_predict.flatten())})
        result = pd.DataFrame({"Actual": list(np.array(self.y_test_data()).flatten()), "Predicted": list(y_predict.flatten())})
#        
#        plt.scatter(range(len(list(np.array(self.y_test_data()).flatten()))), list(np.array(self.y_test_data()).flatten()))
#        plt.plot(range(len(list(np.array(y_predict).flatten()))), list(np.array(y_predict).flatten()))

        return result
    
    def heatmap(self):
        
        sns.heatmap(self.dataset, xticklabels = self.dataset[self.x].corr(), yticklabels = self.dataset[self.x].corr(), cmap = "magma")
    
    def model(self):
        
        regr_data = self.mult_lin_regr()
        
        regr_data["STD Previous 5 Years"] =float(self.clean_data()[self.y].apply(np.std))
        regr_data["Nominal Difference"] = regr_data["Predicted"] - regr_data["Actual"]
        regr_data["STD of Difference"] = np.sqrt(((regr_data["Actual"] - regr_data["Predicted"])**2)/2)
        regr_data["Coefficient"] = regr_data["STD of Difference"]/regr_data["STD Previous 5 Years"]
        
        return regr_data
    
    def model_test(self, start):
            
        model = pd.DataFrame()
        
        
        for i in range(len(range(start,end))):
            
            model = model.append(self.model())
            start += 1
            end += 1
        
        return model
        
    
class Test(Data): 
    
    def __init__(self, dataset_class):
        
        self.dataset_class = dataset_class
    
    
    def model_test(self, start, num_trials):
        
        model = pd.DataFrame()
        years = []
        
        for i in range(num_trials):
            
            frame = MultiRegression(self.dataset_class, ["GVA","FTSE", 'RD Expenditure'],\
                                            ['Employment'], \
                                            start, start+5)  
            years.append(start+6)
            model = model.append(frame.model())
            start += 1
        
#        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#            
#            print(model)
        model["Year"] = years
        col_list = ['Year', 'Actual', 'Predicted', 'STD Previous 5 Years', 'Nominal Difference',
       'STD of Difference', 'Coefficient']
        
        model = model.reindex(columns = col_list)
        
        
        return model
    
        
        

'''Multi Regr Data'''
df_retail = pd.read_excel("MultiRegrData.xlsx", sheet_name = "Retail")
df_admin = pd.read_excel("MultiRegrData.xlsx", sheet_name = "Admin")
df_science = pd.read_excel("MultiRegrData.xlsx", sheet_name = "Science")
df_manuf = pd.read_excel("MultiRegrData.xlsx", sheet_name = "Manufacturing")


'''Multiple Regression Retail'''
mult_retail = MultiRegression(df_retail, ["GVA","FTSE", 'RD Expenditure'],['Employment'], 2010,2015)
#print(mult_retail.clean_data())
#print(mult_retail.corr_table())
#print(mult_retail.y_test_data())

#'''Multiple Regression Admin'''
mult_admin = MultiRegression(df_admin, ["GVA", "FTSE", 'RD Expenditure'],['Employment'], 2010,2015)
#
#'''Multiple Regression Science'''
mult_science = MultiRegression(df_science, ["GVA", "FTSE", 'RD Expenditure'],['Employment'], 2010,2015)

'''Multiple Regression Manufacturing'''
mult_manuf = MultiRegression(df_manuf, ["GVA", "FTSE", 'RD Expenditure'],['Employment'], 2001,2017)

#'''Multi Regr Change Data'''
#df_retail_change = pd.read_excel("MultiRegrData_change.xlsx", sheet_name = "Retail")
#mult_retail_change = MultiRegression(df_retail_change, ['GVA Change', 'FTSE Change', 'RD Expenditure Change'],['Employment Change'], 2001,2015)


test_retail = Test(df_retail)
test_admin = Test(df_admin)
test_science = Test(df_science)
test_manuf = Test(df_manuf)
model_test_retail = test_retail.model_test(2005, 8)
model_test_admin = test_admin.model_test(2005, 8)
model_test_science = test_science.model_test(2005, 8)
model_test_manuf = test_manuf.model_test(2005, 8)
#print(model_test_manuf[["Year", "STD Previous 5 Years", "STD of Difference"]])
print(model_test_manuf)
#print(len(mult_manuf.model().columns))

plt.figure()
test_retail.graph_compare("Model Coefficients Graph", "Year", model_test_retail["Year"], ["Retail", "Administrative", "Scientific Research", "Manufacturing"], \
[model_test_retail["Coefficient"], model_test_admin["Coefficient"], model_test_science["Coefficient"], model_test_manuf["Coefficient"] ])
plt.plot(model_test_retail["Year"], [2]*len(model_test_retail["Year"]), linestyle = "--")
plt.ylabel("Coefficients", fontsize = 10)
plt.savefig("Coefficients_Graph.pdf")

#print("Retail")
#print("-" *27)
#print(mult_retail.mult_lin_regr())
#print("-" *27)
#print("Admin")
#print("-" *27)
#print(mult_admin.mult_lin_regr())
#print("-" *27)
#print("Science")
#print("-" *27)
#print(mult_science.mult_lin_regr())
#print("-" *27)
#print(mult_retail.clean_data().isnull())
#print(mult_retail.clean_data())
#print(mult_retail.graph_table(['2016','2017'],['A', 'B'], [[100,200],[300,400]]))

#print(" "*30,"Retail")
#print("-" *62)
#print(mult_retail.corr_table())
#print("-" *62)
#print(" "*30,"Admin")
#print("-" *62)
#print(mult_admin.corr_table())
#print("-" *62)
#print(" "*30, "Science")
#print("-" *62)
#print(mult_science.corr_table())
#print("-" *62)
#print(" "*30, "Manufacturing")
#print("-" *62)
#print(mult_manuf.corr_table())
#print("-" *62)



#with pd.ExcelWriter('Corr_Tables.xlsx') as writer:
###    
#    mult_retail.corr_table().to_excel(writer, sheet_name ='Retail')
#    mult_admin.corr_table().to_excel(writer, sheet_name ='Admin')
#    mult_science.corr_table().to_excel(writer, sheet_name ='Science')
#    mult_manuf.corr_table().to_excel(writer, sheet_name ='Manufacturing')
    

#with pd.ExcelWriter('Test_Model.xlsx') as writer:
#    
#    test_retail.model_test(2005, 7).to_excel(writer, sheet_name ='Retail')
#    test_admin.model_test(2005, 7).to_excel(writer, sheet_name ='Admin')
#    test_science.model_test(2005, 7).to_excel(writer, sheet_name ='Science')
#    test_manuf.model_test(2005, 7).to_excel(writer, sheet_name ='Manufacturing')

#'''Retail GVA v Employment Correlation'''
retail_gva_empl = SimpleRegression("GVA", "Employment", gva_retail.getY(), empl_retail.getY())
#retail_gva_empl.corr_graph_line(retail_gva_empl.polynomial_regr(1))  
##
##'''Administrative GVA v Employment Correlation'''
admin_gva_empl = SimpleRegression("GVA", "Employment", gva_admin.getY(), empl_admin.getY())
#admin_gva_empl.corr_graph_line(admin_gva_empl.polynomial_regr(1))
#
##'''Scientific Research GVA v Employment Correlation'''
science_gva_empl = SimpleRegression("GVA", "Employment", gva_science.getY(), empl_science.getY())
#science_gva_empl.corr_graph_line(science_gva_empl.polynomial_regr(1))
#

'''Manufacturing GVA v Employment Correlation'''
#plt.subplot(131)
manuf_gva_empl = SimpleRegression("GVA", "Employment", gva_manuf.getY(), empl_manuf.getY())
#manuf_gva_empl.corr_graph_line(manuf_gva_empl.polynomial_regr(1))
#manuf_gva_empl.corr_graph_line(manuf_gva_empl.expo_regr())


'''Manufacturing GVA v Employment Correlation'''
manuf_gva_empl_norm = SimpleRegression("GVA", "Employment", gva_manuf.normalizeY(), empl_manuf.normalizeY())
#print(manuf_gva_empl_norm.expo_regr())
#manuf_gva_empl_norm.corr_graph_line(manuf_gva_empl_norm.expo_regr())
#print(manuf_gva_empl_norm.getY())
#print(manuf_gva_empl_norm.getX())


###'''Retail Employment v FTSE Correlation'''
#retail_empl_ftse = SimpleRegression("FTSE", "Employment", ftse.getY(), empl_retail.getY())
##retail_empl_ftse .corr_graph_line(retail_empl_ftse.polynomial_regr(1)) 
##
###'''Admin Employment v FTSE Correlation'''
#admin_empl_ftse = SimpleRegression("FTSE", "Employment", ftse.getY(), empl_admin.getY())
##admin_empl_ftse.corr_graph_line(admin_empl_ftse.polynomial_regr(1)) 
##
##'''Scientific Research Employment v FTSE Correlation'''
#science_empl_ftse = SimpleRegression("FTSE", "Employment", ftse.getY(), empl_science.getY())
##science_empl_ftse.corr_graph_line(science_empl_ftse.polynomial_regr(1))
#
##'''Retail GVA v FTSE Correlation'''
#retail_gva_ftse = SimpleRegression("FTSE", "GVA ", ftse.getY(), gva_retail.getY())
##retail_gva_ftse .corr_graph_line(retail_gva_ftse.polynomial_regr(1))  
##
##'''Administrative GVA v FTSE Correlation'''
#admin_gva_ftse = SimpleRegression("FTSE", "GVA", ftse.getY(), gva_admin.getY())
##admin_gva_ftse.corr_graph_line(admin_gva_ftse.polynomial_regr(1))
#
##'''Scientific Research GVA v FTSE Correlation'''
#science_gva_ftse = SimpleRegression("FTSE", "GVA", ftse.getY(), gva_science.getY())
##science_gva_ftse.corr_graph_line(science_gva_ftse.polynomial_regr(1)) 

'''Manufacturing Employment v FTSE Correlation'''
#plt.subplot(132)
#manuf_empl_ftse = SimpleRegression("FTSE", "Employment", ftse.getY(), empl_manuf.getY())
#manuf_empl_ftse.corr_graph_line(manuf_empl_ftse.polynomial_regr(1))

'''Manufacturing Employment v FTSE Normalised Correlation'''
manuf_empl_ftse_norm = SimpleRegression("FTSE", "Employment", ftse.normalizeY(), empl_manuf.normalizeY())
#manuf_empl_ftse_norm.corr_graph_line(manuf_empl_ftse_norm.expo_regr())

#
##'''Retail RD v Employment Correlation'''
#retail_rd_empl = SimpleRegression("R&D Expenditure", "Employment", rd_retail.getY(), empl_retail.getY())
##retail_rd_empl.corr_graph_line(retail_rd_empl.polynomial_regr(1))  
##
##'''Administrative RD v Employment Correlation'''
#admin_rd_empl = SimpleRegression("R&D Expenditure", "Employment", rd_admin.getY(), empl_admin.getY())
##admin_rd_empl.corr_graph_line(admin_rd_empl.polynomial_regr(1))
##
##'''Scientific Research RD v Employment Correlation'''
#science_rd_empl = SimpleRegression("R&D Expenditure", "Employment", rd_science.getY(), empl_science.getY())
##science_rd_empl.corr_graph_line(science_rd_empl.polynomial_regr(1))
#
#'''Manufacturing RD v Employment Correlation'''
##plt.subplot(133)
manuf_rd_empl = SimpleRegression("R&D Expenditure", "Employment", rd_manuf.getY(), empl_manuf.getY())
#manuf_rd_empl.corr_graph_line(manuf_rd_empl.polynomial_regr(1))

'''Manufacturing RD v Employment Normalised Correlation'''
manuf_rd_empl_norm = SimpleRegression("R&D Expenditure", "Employment", rd_manuf.normalizeY(), empl_manuf.normalizeY())
#manuf_rd_empl.corr_graph_line(manuf_rd_empl.polynomial_regr(1))
#manuf_rd_empl_norm.corr_graph_line(manuf_rd_empl_norm.expo_regr())




def correlation_graphs(fig_title, gva_regr, ftse_regr, rd_regr):
    
    '''Prints a figure with all correlations as subplots 
    Industry: string that defines the industry 
    Fig_title: the title of the figure
    '''

    plt.figure(figsize = [18,5])
    plt.subplot(131)
    gva_regr.corr_graph_line(gva_regr.polynomial_regr(1))
    plt.subplot(132)
    ftse_regr.corr_graph_line(ftse_regr.polynomial_regr(1))
    plt.subplot(133)
    rd_regr.corr_graph_line(rd_regr.polynomial_regr(1))
    plt.suptitle(fig_title, fontsize=20, fontweight = "bold")
    plt.subplots_adjust(left=None, bottom=None, right=None, top= 0.8, wspace= 0.28, hspace= 10)

def multiple_graphs(fig_title):
    
    '''Prints a figure with all correlations as subplots 
    Industry: string that defines the industry 
    Fig_title: the title of the figure
    '''

    plt.figure(figsize = [20,5])
    plt.subplot(131)
    gva_retail.graph_compare("GVA by Sector", "Year", gva_retail.getX(), ["Retail", "Administrative", "Scientific Research", "Manufacturing"], \
    [gva_retail.normalizeY(), gva_admin.normalizeY(), gva_science.normalizeY(), gva_manuf.normalizeY() ])
    plt.subplot(132)
    empl_retail.graph_compare("Employment by Sector", "Year", empl_retail.getX(), ["Retail", "Administrative", "Scientific Research", "Manufacturing" ], \
    [empl_retail.normalizeY(), empl_admin.normalizeY(), empl_science.normalizeY(), empl_manuf.normalizeY()])
    plt.subplot(133)
    rd_retail.graph_compare("R&D by Sector", "Year", rd_retail.getX(), ["Retail", "Administrative", "Scientific Research", "Manufacturing"], \
    [rd_retail.normalizeY(), rd_admin.normalizeY(), rd_science.normalizeY(), rd_manuf.normalizeY()])
    plt.suptitle(fig_title, fontsize=20, fontweight = "bold")
    plt.subplots_adjust(left=None, bottom=None, right=None, top= 0.8, wspace= 0.3, hspace= 10)


#multiple_graphs("Normalised Graphs")
#plt.savefig("NormalisedGraphs.pdf")
#with PdfPages("CorrelationGraphs.pdf") as corrs:
#    corrs.savefig(correlation_graphs("Retail", retail_gva_empl, retail_empl_ftse, retail_rd_empl))
#    corrs.savefig(correlation_graphs("Administration Services",admin_gva_empl, admin_empl_ftse, admin_rd_empl))
#    corrs.savefig(correlation_graphs("Science",science_gva_empl, science_empl_ftse, science_rd_empl))
#    corrs.savefig(correlation_graphs("Manufacturing",manuf_gva_empl, manuf_empl_ftse, manuf_rd_empl))
    




































