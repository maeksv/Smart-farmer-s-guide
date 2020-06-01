# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 00:29:02 2019

@author: Hp
"""
print ("             SMART FARMER'S ASSISTANT     ")
print ("                     welcome ! ")
print( "enter your name below :  ")
name = input()
print("-------------------------------------------------------------")
print( "                        hello "+name )
print(" ENTER DETAILS ")
aa=input("crop season 1. kharif 2. rabi 3. zaid : ")
a1=input("min price :")
a2=input("max price:")


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
dataset=pd.read_csv('pred1.csv')
x= dataset.iloc[:,[1,3,5,6]].values
y= dataset.iloc[:,[8]].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:,1] = labelencoder_X.fit_transform(x[:, 1])
labelencoder_X = LabelEncoder()
x[:,0] = labelencoder_X.fit_transform(x[:, 0])
x = np.asarray(x,dtype='int64')

#encoding categorical data with onehotencoder 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:,0] = labelencoder_X.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
x=x[:,[1,2,3,4,5]]
#x=x[:,[0,2,3]]
#splitting the dataset into training set and test set 
from sklearn.model_selection  import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size=0.2)


#fitting  multilinear regression 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train )

#predicting the test set results
y_pred = regressor.predict(x_test)

#backward elimination
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((31,1)).astype(int), values = x ,axis =1)

import statsmodels.api as sm
x_opt = x[:, [0,2,4,5]]
regressor_OLS = sm.OLS(endog =y ,exog = x_opt).fit()
regressor_OLS.summary()


#print('Variance score: %.2f' % regressor.score(x_test, y_test))

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
accuracy = regressor.score(x_test,y_test)
#print(accuracy*100,'%')





# importing pandas package 
import pandas as pd 
dataset=pd.read_csv('pred1.csv')
  
# replacing blank spaces with '_'  
dataset.columns =[column.replace(" ", "_") for column in dataset.columns] 
  
# filtering with query method 
if(aa==1):
    dataset.query('season_ =="kharif" ', inplace = True) 
elif(aa==2):
    dataset.query('season_ == "rabi"', inplace = True) 
else:
    dataset.query('season_ == "zaid"', inplace = True) 
    
dataset.query('irrigation_ == "y"', inplace = True) 
dataset.query('irrigation_ == "n"', inplace = True) 
dataset.query('cost_ >= "2000"', inplace = True) 
# display 
dataset 


 #region wise crop 
 # using weather forecasting on crop 
 # crop stages 
 #sale price should be givt 
 