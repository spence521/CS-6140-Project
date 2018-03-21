#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 12:07:48 2018

@author: mattiamedinagrespan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
 



dataset = pd.read_csv('student_scores.csv')  
dataset1=pd.read_csv('petrol_consumption.csv')

newdata=pd.read_csv('New_Data.csv')

X = newdata.iloc[0:3,0]
#print(X) 
#y = dataset.iloc[:, 1].values 

#print(dataset.shape)

# =============================================================================
# #print(newdata.shape)
# print(newdata.head())
# 
# newdata.plot(x='Budget', y='Vote_Average', style='o')  
# plt.title('Hours vs Percentage')  
# plt.xlabel('Hours Studied')  
# plt.ylabel('Percentage Score')  
# plt.show()  
# 
# 
# X = newdata[['Budget']]
# y = newdata['Vote_Average'] 
# 
# #print(newdata.iloc[:,:])
# #print(X)
# #J = newdata.iloc[:,0].values
# #print(J)
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
# 
# #print(X_test)
# #print(y_test)
# 
# regressor = LinearRegression()  
# regressor.fit(X_train, y_train) 
# 
# print(regressor.intercept_) 
# 
# print(regressor.coef_)  
# 
# y_pred = regressor.predict(X_test) 
# 
# 
# df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
# print(df) 
# =============================================================================


def twodReg(data,xaxis,yaxis):
    newdata.plot(x=xaxis, y=yaxis, kind='scatter')  
    #plt.title('Hours vs Percentage')  
    plt.xlabel(xaxis)  
    plt.ylabel(yaxis)  
    plt.show()  
    X = newdata[[xaxis]]
    y = newdata[yaxis] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()  
    regressor.fit(X_train, y_train) 
    y_pred = regressor.predict(X_test)
    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
    #print(df) 
    return 'f'
  
# =============================================================================
# twodReg(newdata,'Budget','Vote_Average') 
# twodReg(newdata,'Woman_to_Total_Ratio','Vote_Average')   
# twodReg(newdata,'Woman_to_Total_Ratio','Budget') 
# twodReg(newdata,'Woman_to_Total_Ratio','Revenue') 
# twodReg(newdata,'Woman_to_Total_Ratio','Revenue_to_Budget_Ratio') 
# twodReg(newdata,'Woman_to_Total_Ratio','Popularity') 
# =============================================================================

# =============================================================================
# print(dataset1.shape)
# 
# print(dataset1.head())
# 
# print(dataset1.describe())
# =============================================================================

# =============================================================================
# X = dataset1[['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']]
# y = dataset1['Petrol_Consumption']  
# 
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# 
# 
# regressor = LinearRegression()  
# regressor.fit(X_train, y_train) 
# 
# coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])  
# print(coeff_df)
# 
# y_pred = regressor.predict(X_test) 
# 
# df = pd.DataFrame({'Act': y_test, 'Predicted': y_pred})  
# print(df) 
# 
# =============================================================================
#print(dataset.describe())

#print(dataset)
# =============================================================================
# dataset.plot(x='Hours', y='Scores', style='o')  
# plt.title('Hours vs Percentage')  
# plt.xlabel('Hours Studied')  
# plt.ylabel('Percentage Score')  
# plt.show()  
# =============================================================================

#print(dataset)
# =============================================================================
# X = dataset.iloc[:, :-1].values  
# y = dataset.iloc[:, 1].values 
# 
# #print(X)
# J = dataset.iloc[:,0].values
# print(J)
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
# 
# print(X_test)
# print(y_test)
# 
# regressor = LinearRegression()  
# regressor.fit(X_train, y_train) 
# 
# print(regressor.intercept_) 
# 
# print(regressor.coef_)  
# 
# y_pred = regressor.predict(X_test) 
# 
# 
# df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
# print(df) 
# =============================================================================


#import scipy.interpolate as sci 

# =============================================================================
# x=np.array([0,1,2,3,4,5])
# y=np.array([0,0.8,0.9,0.1,-0.8,-1])
# 
# print(x)
# print(y)
# 
# p1=np.polyfit(x,y,15)
# 
# print(np.polyval(p1,15))
# print(p1)
# plt.plot(x,y,'o')
# plt.plot(x,np.polyval(p1,x))
# =============================================================================


def readFileData(fileName):
    data = open(fileName).readlines()

    #return data.strip('\n').split('\n')
    return data

def stripData(doc):
    vector=[]
    for line in doc[1:]:
        vector.append(line.strip('\n').split(','))
    return np.array(vector).astype(np.float)

doc=readFileData('New_Data.csv')

docu=stripData(doc)


def phi(point, cent):
        dist=1000000000.0
        c=[]
        for center in cent:
            if np.linalg.norm(point-center)<dist:
                dist=np.linalg.norm(point-center)
                c=center
        return c
    
def initialize(X, K):
    C = [X[0]]
    for k in range(1, K):
        D2 = np.array([(np.linalg.norm(x-phi(x, C)))**2 for x in X])
        probs = D2/D2.sum()
        cumprobs = probs.cumsum()
        r = np.random.rand()
        for j,p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        C.append(X[i])
    return C


print(np.linalg.norm(docu[0]-docu[1]))
initialize(docu,3)



#print(np.concatenate((docu[0][0:3],docu[0][10:12]),axis=0))



#print(doc[1])

#print(np.array(doc[1].split(',')).astype(np.float))

#print(stripData(doc[1]))

#a=[ex[1:] for ex in stripData(doc)]

