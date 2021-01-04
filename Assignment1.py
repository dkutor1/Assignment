import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class ExploratoryDataAnalysis:
    def loadDataRawHouseData():
        data = pd.read_csv("C:/Users/kutor/Desktop/UF/EnhancedIT/Assignments/AS_1/raw_house_data.csv",
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
    def deleteMissingRows():
        df = ExploratoryDataAnalysis.loadDataRawHouseData()
        df = df[df.fireplaces != " "]
        df = df[df.lot_acres.notnull()]
        #df = df[df.lot_acres != 0]
        return df

    def inputateNoneValues():
        #bathrooms with mean
        df = ExploratoryDataAnalysis.deleteMissingRows()
        tempDf = df[df.bathrooms != "None"]
        bathrooms = pd.to_numeric(tempDf.bathrooms) #convert object to float
        bathrooms = np.array(bathrooms)
        bathroomsMean = int(np.mean(bathrooms))
        df.loc[df["bathrooms"] == "None", "bathrooms"] = bathroomsMean

        
        #garage with zero
        df.loc[df["garage"] == "None", "garage"] = 0

        #HOA with zero
        df.loc[df["HOA"] == "None", "HOA"] = 0
        #df.loc[df["HOA"] == "20,000", "HOA"] = 20000

        #sqrt_ft with mean
        tempDf = df[df.sqrt_ft != "None"]
        sqrt_ft = pd.to_numeric(tempDf.sqrt_ft) #convert object to float
        sqrt_ft = np.array(sqrt_ft)
        sqrt_ftMean = np.mean(sqrt_ft)
        df.loc[df["sqrt_ft"] == "None", "sqrt_ft"] = sqrt_ftMean
        
        return df
        

    def cleanedDataSet():
        return ExploratoryDataAnalysis.inputateNoneValues()

    def plotData():
        
        df = ExploratoryDataAnalysis.cleanedDataSet()

        quantData = np.array([df.MLS, df.sold_price, df.zipcode, df.longitude, df.latitude,
                              df.lot_acres, df.taxes, df.year_built, df.bedrooms, df.bathrooms,
                              df.sqrt_ft, df.garage, df.fireplaces])
        
        quantTitle = np.array(["MLS", "sold_price", "zipcode", "longitude", "latitude",
                              "lot_acres", "taxes", "year_built", "bedrooms", "bathrooms",
                              "sqrt_ft", "garage", "fireplaces"])
        
        qualData = np.array([df.kitchen_features, df.floor_covering])

        #boxplot and whisker of sold_price
        for i in range(len(quantData)):
            plt.figure(figsize=(4,3))
            plt.boxplot(pd.to_numeric(quantData[i]))
            plt.title(quantTitle[i])
            plt.show()

        quantDataOthers = np.array([df.MLS, df.zipcode, df.longitude, df.latitude,
                              df.lot_acres, df.taxes, df.year_built, df.bedrooms, pd.to_numeric(df.bathrooms),
                              pd.to_numeric(df.sqrt_ft), pd.to_numeric(df.garage), df.fireplaces])

        quantTitleOthers = np.array(["MLS vs sold_price", "zipcode vs sold_price", "longitude vs sold_price",
                                     "latitude vs sold_price", "lot_acres vs sold_price", "taxes vs sold_price",
                                     "year_built vs sold_price", "bedrooms vs sold_price", "bathrooms vs sold_price",
                                     "sqrt_ft vs sold_price", "garage vs sold_price", "fireplaces vs sold_price"])
        
        #sold_price vrs all quantitative data
        for i in range(len(quantDataOthers)):
            #plt.subplot()
            plt.figure(figsize=(4,3))
            plt.scatter(quantDataOthers[i], df.sold_price)
            plt.title(quantTitleOthers[i])
            plt.show()
            
                              
    
        

        
