# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:07:29 2021

@author: Field Employee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

class ExploratoryDataAnalysis():
    def loadDataRawHouseData(self):
        data = pd.read_csv("C:/Users/Field Employee/.spyder-py3/AS_1/raw_house_data.csv",
                           sep = ",", header = 0)
        pd.set_option('display.max_columns', 0)
        df = pd.DataFrame(data)
        return df
    
    
    def describeRawHouseData():
        df = ExploratoryDataAnalysis.loadDataRawHouseData()
        #print(df.describe())
        pd.set_option('display.max_columns', None)
        return df.describe()
    
    
    def findMissingValues():
        df = ExploratoryDataAnalysis.loadDataRawHouseData()
        missing = df.isnull()
        rowColumnMiss = np.where(missing)
        return rowColumnMiss
    
    
    def percentMissing():
        df = ExploratoryDataAnalysis.loadDataRawHouseData()
        fplaces = df.fireplaces
        fplacesMissing = fplaces[fplaces == " "]
        missing = ExploratoryDataAnalysis.findMissingValues()
        
        rows = missing[0]
        totalMissing = len(rows) + len(fplacesMissing)
        df = ExploratoryDataAnalysis.loadDataRawHouseData()
        totalRows = len(df)
        return "{:.2%}".format(totalMissing/totalRows)
    
    
    #1.delete lot_acres missing rows, 2.delete fireplaces missing rows
    def deleteMissingRows(self):
        df = ExploratoryDataAnalysis().loadDataRawHouseData()
        df = df[df.fireplaces != " "]
        df = df[df.lot_acres.notnull()]
        #df = df[df.lot_acres != 0]
        return df
    
    
    def inputateNoneValues():
        #bathrooms with mean
        df = ExploratoryDataAnalysis().deleteMissingRows()
        tempDf = df[df.bathrooms != "None"]
        bathrooms = pd.to_numeric(tempDf.bathrooms) #convert object to float
        bathrooms = np.array(bathrooms)
        bathroomsMean = int(np.mean(bathrooms))
        df.loc[df["bathrooms"] == "None", "bathrooms"] = bathroomsMean
        bathrooms = pd.to_numeric(df.bathrooms)
        df['bathrooms'] = bathrooms
        
        #garage with zero
        df.loc[df["garage"] == "None", "garage"] = 0
        #HOA with zero
        df.loc[df["HOA"] == "None", "HOA"] = 0
        hoa = np.array(df.HOA)
        hoa = hoa.astype(str)
        for i in range(len(hoa)):
            hoa[i] = hoa[i].replace(',', '')
        
        hoa = pd.to_numeric(hoa)
        
        for i in range(len(hoa)):
            if (hoa[i] == 0):
                hoa[i] = 0
            else:
                hoa[i] = 1
        hoa = hoa.astype(int)
        df['HOA'] = hoa
        
        
        #sqrt_ft with mean
        tempDf = df[df.sqrt_ft != "None"]
        sqrt_ft = pd.to_numeric(tempDf.sqrt_ft) #convert object to float
        sqrt_ft = np.array(sqrt_ft)
        sqrt_ftMean = np.mean(sqrt_ft)
        df.loc[df["sqrt_ft"] == "None", "sqrt_ft"] = sqrt_ftMean
        
        df = df[(df.bedrooms != 36)] 
        df = df[(df.bedrooms != 18)]
        df = df[(df.bedrooms != 19)]
        df = df[(df.bedrooms != 13)]
        
        df = df[(df.bathrooms.astype(int) != 36)]
        df = df[(df.bathrooms.astype(int) != 11)]
        baths = df.bathrooms.astype(int)
        df['bathrooms'] = baths
        
        df['price_per_sqft'] = df.sold_price/pd.to_numeric(df.sqrt_ft)
        return df
    
        
    def cleanedDataSet(self):
        return ExploratoryDataAnalysis.inputateNoneValues()
    
    
    def processXandY(self):
        df = ExploratoryDataAnalysis.inputateNoneValues()
        
        #self.y = np.array(df.bedrooms)
        #self.y = np.array(df.bathrooms)
        self.y = np.array(df.HOA)
        
        #[df.zipcode, df.longitude, df.latitude, df.year_built,df.lot_acres, df.taxes, df.bedrooms, pd.to_numeric(df.bathrooms),
                              #pd.to_numeric(df.sqrt_ft), pd.to_numeric(df.garage), pd.to_numeric(df.fireplaces)]
        
        self.X = np.array([pd.to_numeric(df.longitude), pd.to_numeric(df.latitude)])
        self.X = self.X.T
        N, D = self.X.shape
        #for i in range(D):
              #self.X[:, i] = self.X[:, i]/(np.amax(self.X[:, i]) - np.amin(self.X[:, i]))
        
        #self.y = self.y/(np.amax(self.y) - np.amin(self.y))
        return self.X, self.y
    
    
    def plotData(self):
        
        df = ExploratoryDataAnalysis().cleanedDataSet()
        quantData = np.array([df.MLS, df.sold_price, df.zipcode, df.longitude, df.latitude,
                              df.lot_acres, df.taxes, df.year_built, df.bedrooms, df.bathrooms,
                              df.sqrt_ft, df.garage, df.fireplaces])
        
        quantTitle = np.array(["MLS", "sold_price", "zipcode", "longitude", "latitude",
                              "lot_acres", "taxes", "year_built", "bedrooms", "bathrooms",
                              "sqrt_ft", "garage", "fireplaces"])
        
        qualData = np.array([df.kitchen_features, df.floor_covering])
        #boxplot and whisker of sold_price
        # for i in range(len(quantData)):
        #     plt.figure(figsize=(4,3))
        #     plt.boxplot(pd.to_numeric(quantData[i]))
        #     plt.title(quantTitle[i])
        #     plt.show()
        quantDataOthers = np.array([df.MLS, df.zipcode, df.longitude, df.latitude,
                              df.lot_acres, df.taxes, df.year_built, df.bedrooms, pd.to_numeric(df.bathrooms),
                              pd.to_numeric(df.sqrt_ft), pd.to_numeric(df.garage), df.fireplaces])
        quantTitleOthers = np.array(["MLS vs sold_price", "zipcode vs sold_price", "longitude vs sold_price",
                                     "latitude vs sold_price", "lot_acres vs sold_price", "taxes vs sold_price",
                                     "year_built vs sold_price", "bedrooms vs sold_price", "bathrooms vs sold_price",
                                     "sqrt_ft vs sold_price", "garage vs sold_price", "fireplaces vs sold_price"])
        
        #sold_price vrs all quantitative data
        # for i in range(len(quantDataOthers)):
        #     #plt.subplot()
        #     plt.figure()
        #     plt.scatter(quantDataOthers[i], df.price_per_sqft)
        #     plt.title(quantTitleOthers[i])
        #     plt.show()
            
        
        #bedrooms
        fig = plt.figure(figsize = (15,10))
        ax = fig.add_subplot(111)
        #cax = ax.scatter(pd.to_numeric(df.sqrt_ft), df.price_per_sqft , c = df.bedrooms, cmap='tab20c')
        cax = ax.scatter(df.price_per_sqft , pd.to_numeric(df.sqrt_ft),  c = df.bedrooms, cmap='tab20c')
        #plt.xlim(-1,30)
        #plt.ylim(-1,15000)
        plt.xlabel("sqrt_ft")
        plt.ylabel("price_per_sqft")
        plt.title("Bedrooms based on sqrt_ft and price_per_sqft")
        fig.colorbar(cax)
        plt.show()    
        
        
        #bathrooms
        fig = plt.figure(figsize = (15,10))
        ax = fig.add_subplot(111)
        #cax = ax.scatter(pd.to_numeric(df.sqrt_ft), df.price_per_sqft, c = df.bathrooms, cmap='tab20c')
        cax = ax.scatter(df.price_per_sqft, pd.to_numeric(df.sqrt_ft),  c = df.bathrooms, cmap='tab20c')
        #plt.xlim(-1,30)
        #plt.ylim(-1,15000)
        plt.xlabel("sqrt_ft")
        plt.ylabel("price_per_sqft")
        plt.title("Bathrooms based on sqrt_ft and price_per_sqft")
        fig.colorbar(cax)
        plt.show()
        
        
        #HOA
        fig = plt.figure(figsize = (15,10))
        ax = fig.add_subplot(111)
        #cax = ax.scatter(pd.to_numeric(df.sqrt_ft), df.price_per_sqft, c = df.bathrooms, cmap='tab20c')
        cax = ax.scatter(df.price_per_sqft, pd.to_numeric(df.sqrt_ft),  c = df.HOA, cmap='tab20c')
        #plt.xlim(-1,30)
        #plt.ylim(-1,15000)
        plt.xlabel("sqrt_ft")
        plt.ylabel("price_per_sqft")
        plt.title("HOA based on sqrt_ft and price_per_sqft")
        fig.colorbar(cax)
        plt.show()
        
        
        fig = plt.figure(figsize = (15,10))
        ax = fig.add_subplot(111)
        #cax = ax.scatter(pd.to_numeric(df.sqrt_ft), df.price_per_sqft, c = df.bathrooms, cmap='tab20c')
        cax = ax.scatter(df.bedrooms, df.bathrooms)#,   cmap='tab20c') #c = df.sold_price,
        #plt.xlim(-1,30)
        #plt.ylim(-1,15000)
        plt.xlabel("bedrooms")
        plt.ylabel("bathrooms")
        plt.title("bedrooms vs bathrooms")
        #fig.colorbar(cax)
        plt.show()

    def corrMatrix(self):
        eda = ExploratoryDataAnalysis()
        df = eda.cleanedDataSet()
        pd.set_option('display.max_columns', None)
        return df.corr()
    
    
    def heatMapMatrix(self):
        eda = ExploratoryDataAnalysis()
        df = eda.cleanedDataSet()
        ax = plt.axes()
        sn.heatmap(df.corr(), cmap='Reds', ax = ax)
        ax.set_title('Correlation Matrix Between Variables')
        plt.show()
    
    def plotAccuracy(self):
       X = ["Bedrooms", "Bathrooms", "HOA"]
       Y = [81.268882, 86.102719, 69.889225]
       plt.bar(X, Y, color = ['g', 'r', 'b'])
       plt.xlabel("Features")
       plt.ylabel("Percent Accuracy")
       plt.title("Model Performance")
       plt.show()

