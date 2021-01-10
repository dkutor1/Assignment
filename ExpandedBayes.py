# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 12:51:43 2021

@author: Field Employee
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import multivariate_normal as mvn
from scipy.stats import multinomial as mlvn
from scipy.stats import bernoulli as brn
#matplotlib inline
class processData:
    def loadTestData(self):
        data =  pd.read_csv("C:/Users/Field Employee/.spyder-py3/AS_2/MNIST_test.csv", sep=",", header = 1)
        df = pd.DataFrame(data)
        df = df.to_numpy()
        self.Xtest = df[:, 3:]/255
        self.ytest = df[:, 2]
        
    def loadTrainData(self):
        data = pd.read_csv("C:/Users/Field Employee/.spyder-py3/AS_2/MNIST_train.csv", sep=",", header = 1)
        df = pd.DataFrame(data)
        df = df.to_numpy()
        self.X = df[:, 3:]/255
        self.y = df[:, 2]
        #return df
    
        
class GaussNB():
    def fit(self, X, y, epsilon = 1e-3):
        self.likelihoods = dict()
        self.priors = dict()
        
        self.K = set(y.astype(int))
        
        for k in self.K:
            X_k = X[y == k,:]
            self.likelihoods[k] = {"mean":X_k.mean(axis=0), "cov":X_k.var(axis=0) + epsilon}
            self.priors[k] = len(X_k)/len(X)
            
    def predict(self, X):
        N, D = X.shape
        
        P_hat = np.zeros((N,len(self.K)))
        
        for k, l in self.likelihoods.items():
            P_hat[:,k] = mvn.logpdf(X, l["mean"], l["cov"]) + np.log(self.priors[k])
        return P_hat.argmax(axis = 1)


class GaussBayes():
    def fit(self, X, y, epsilon = 1e-3):
        self.likelihoods = dict()
        self.priors = dict()
        
        self.K = set(y.astype(int))
        
        for k in self.K:
            X_k = X[y == k,:]
            N_k, D = X_k.shape
            mu_k=X_k.mean(axis=0)
            self.likelihoods[k] = {"mean":X_k.mean(axis=0), "cov":(1/(N_k-1))*np.matmul((X_k-mu_k).T,X_k-mu_k)+ epsilon*np.identity(D)}
            self.priors[k] = len(X_k)/len(X)
            
    def predict(self, X):
        N, D = X.shape
        
        P_hat = np.zeros((N,len(self.K)))
        
        for k, l in self.likelihoods.items():
            P_hat[:,k] = mvn.logpdf(X, l["mean"], l["cov"]) + np.log(self.priors[k])
            
        return P_hat.argmax(axis = 1)
    
class GenGaussBayes():
    def fit(self, X, y, epsilon = 1e-3):
        self.likelihoods = dict()
        self.priors = dict()
        
        self.K = set(y.astype(int))
        
        for k in self.K:
            X_k = X[y == k,:]
            N_k, D = X_k.shape
            mu_k=X_k.mean(axis=0)
            self.likelihoods[k] = {"mean":X_k.mean(axis=0), "cov":(1/(N_k-1))*np.matmul((X_k-mu_k).T,X_k-mu_k)+ epsilon*np.identity(D)}
            self.priors[k] = len(X_k)/len(X)
            
    def predict(self, X,DistFam):
        N, D = X.shape
        
        P_hat = np.zeros((N,len(self.K)))
        
        for k, l in self.likelihoods.items():
            P_hat[:,k] = DistFam(X, l["mean"], l["cov"]) + np.log(self.priors[k])
            
        return P_hat.argmax(axis = 1)
    
class GenBayes():
    
    def fit(self, X, y, DistStr, epsilon = 1e-3):
        N, D = X.shape
        self.likelihoods = dict()
        self.priors = dict()
        
        self.K = set(y.astype(int))
        if DistStr=="Gauss":
        
            for k in self.K:
                X_k = X[y == k,:]
                N_k, D = X_k.shape
                mu_k=X_k.mean(axis=0)
                self.likelihoods[k] = {"mean":X_k.mean(axis=0), "cov":(1/(N_k-1))*np.matmul((X_k-mu_k).T,X_k-mu_k)+ epsilon*np.identity(D)}
                self.priors[k] = len(X_k)/len(X)
            return
        if DistStr=="Multinomial": #work on this to make it work
            for k in self.K:
                X_k = X[y == k,:]
                N_k, D = X_k.shape
                mu_k=X_k.mean(axis=0)
                self.likelihoods[k] = {"N":N, "P":sum(N_k/len(X))}
                self.priors[k] = len(X_k)/len(X)
            
        if DistStr=="Bernoulli":
            for k in self.K:
                x_k = X[y == k,:]
                N_k, D = x_k.shape
               
                p = (sum(x_k) + 1) / (len(x_k) + 2)
                self.likelihoods[k] = {"mean": p, "cov": p*(1-p) + epsilon}
                self.priors[k] = len(x_k)/len(X)
            
    def predict(self, X,DistStr):
        N, D = X.shape
        
        if DistStr=="Gauss":
            P_hat = np.zeros((N,len(self.K)))

            for k, l in self.likelihoods.items():
                P_hat[:,k] = mvn.logpdf(X, l["mean"], l["cov"]) + np.log(self.priors[k])

            return P_hat.argmax(axis = 1)
        
        if DistStr=="Multinomial":
            P_hat = np.zeros((N,len(self.K)))

            for k, l in self.likelihoods.items():
                P_hat[:,k] = mlvn.logpmf(X, l["N"], l["P"]) + np.log(self.priors[k])

            return P_hat.argmax(axis = 1)

        if DistStr=="Bernoulli":
            P_hat = np.zeros((N,len(self.K)))

            for k, l in self.likelihoods.items():
                #Using the Bernoulli function/formula. Trick is to get the matrices to go from mxn to a 1x1 number for each k value
                P_hat[:, k] = np.log(self.priors[k]) + np.matmul(X, np.log(l["mean"])) + np.matmul((1-X), np.log(abs(1 - l["mean"])))

            return P_hat.argmax(axis = 1)

def accuracy(y_hat, y):
    return np.mean(y == y_hat)

def plotModelResults(result):
    types_ = ["Naive bayes", "Gauss Bayes", "Bernoulli"]
    plt.bar(types_, result, color = ['g', 'r', 'b'], width = 0.5)
    plt.title("Performance of Models in Percentage")
    #plt.legend()
    plt.show()
    
def makeTable(result):
    t = ["Naive bayes", "Gauss Bayes", "Bernoulli"]
    df = pd.DataFrame({t[0]:result[0], t[1]:result[1], t[2]:result[2]}, index = [0])
    return df

def plotIndividualStats(A, B):
    plt.bar(A, B)
    plt.title("Percent Performance of digits under Gauss Bayes")
    plt.show()