# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 08:19:17 2021

@author: Field Employee
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from AS_1 import Assignment1 as as1
import seaborn as sns
import sweetviz as sw

class linearRegression:
    def fit(self, X, y):
        self.w = np.linalg.solve(X.T@X, X.T@y)
        
    def predict(self, X):
        return np.matmul(X, self.w)


class kNNRegressor:
    def fit(self, X, y):
        self.X = X
        self.y = y
    
    def predict(self, X, K, epsilon = 1e-5):
        N = len(X)
        #print(N)
        y_hat = np.zeros(N)
        
        for i in range(N):
            dixt2 = np.sum((self.X - X[i])**2, axis = 1)
            idxt = np.argsort(dixt2)[:K]
            #print(dixt2[idxt])
            #print(np.exp(-dixt2[idxt]).sum())
            gamma_k = np.exp(-dixt2[idxt])/(np.exp(-dixt2[idxt]) + epsilon).sum()
            y_hat[i] = gamma_k.dot(self.y[idxt])
        return y_hat
    
    

class conValDat:
    def create(self, D, N, r = 20):
        self.X = np.linspace(0, r, N).reshape(N, D)
        self.y = np.sqrt(self.X) + np.exp(-(self.X - 5)**2) - 2*(np.exp(-(self.X - 12.5)**2) + np.random.rand(N, 1)*0.2)
        return self.X, self.y
    
    def show(self):
        plt.figure()
        plt.scatter(self.X, self.y)
        
        
#create a new class OurLinearRegression
class OurLinearRegression():
    #create a fit method 
    def fit(self, X, y, epochs = 1e3, eta = 1e-3, show_curve = False, lambd = 0, p = 1):
        
        ols = OurLinearRegression()
        
        epochs = int(epochs) #number of iterations
        N, D = X.shape #get dimensions of X
        Y = y
        
        self.W = np.random.randn(D) #initialize weight to random normal variables 
        J = np.zeros(epochs) #initialize cost to zeros
        
        for epoch in range(epochs):
            Y_hat = self.predict(X)
            J[epoch] = ols.OLS(Y, Y_hat, N) + (lambd/(p*N))*np.linalg.norm(self.W, ord = p, keepdims = True)
            self.W -= eta*(1/N)*(X.T@(Y_hat - Y) + (1/N)*(lambd*np.abs(self.W)**(p-1)*np.sign(self.W)))
            
            if show_curve:
                plt.figure()
                plt.plot(J)
                plt.xlabel("epochs")
                plt.ylabel("$\mathcal{J}$")
                plt.title("Training Curve")
                plt.show()
    
    #create a predict method
    def predict(self, X):
        return X@self.W
    
    #create method OLS
    def OLS(Y, Y_hat, N):
        return (1/(2*N))*np.sum((Y-Y_hat)**2)
    
    #create method R2
    def R2(self, Y, Y_hat):
        
        #print(np.sum((Y - Y_hat)**2))
        #print(np.sum((Y - np.mean(Y))**2))
        
        return 1 - (np.sum((Y - Y_hat)**2)/(np.sum((Y - np.mean(Y))**2)))
        

def trainAndTestData():
    eda = as1.ExploratoryDataAnalysis()
    X, y = eda.processXandY()
    N = len(y)
    
    #Comment out before running for HOA
    # for i in range(N):
    #     if (y[i] <= 2):
    #         y[i] = 0
            
    #     elif (y[i] > 2 and y[i] <= 4):
    #         y[i] = 1
    #     elif (y[i] > 4 and y[i] <= 6):
    #         y[i] = 2
    #     elif (y[i] > 6 and y[i] <= 8):
    #         y[i] = 3
    #     else:
    #         y[i] = 4
    N, D = X.shape
    N1 = int(0.8*N)
    N2 = int(0.2*N)
    
    X_train = X[0:N1, :]
    y_train = y[0:N1]
    
    X_test = X[N1:, :]
    y_test = y[N1:]
    
    
    
    return X_train, y_train, X_test, y_test

def main():
    
    X_train, y_train, X_test, y_test = trainAndTestData()
    
    #eda = as1.ExploratoryDataAnalysis()
    #X, y = eda.processXandY()
    
    
    #myDat = conValDat()
    #X, y = myDat.create(1, 200)
    #myDat.show()
    
    #Use KNN Regression model
    knn = kNNRegressor()
    knn.fit(X_train, y_train)
    y_hat = knn.predict(X_train, 10)
    
    # print(y)
    # print(y_hat)
    # print(y_hat.shape)
    # print(X.shape)
    
    # plt.figure()
    # #plt.scatter(X[:,0], y)
    # plt.plot(X[:,0], y_hat, color = 'g')
    
    # A = np.array([3.5558])
    # A_hat = knn.predict(A, 10)
    # plt.scatter(A, A_hat, color = 'r')
    
    #Use Linear Regression model
    lr = linearRegression()
    lr.fit(X_train, y_train)
    y_hat2 = lr.predict(X_train)
    
    #Use Our Linear Regression model R2
    olr = OurLinearRegression()
    
    print(olr.R2(y_train, y_hat))
    print(olr.R2(y_train, y_hat2))
    
    return  y_hat, y_hat2
