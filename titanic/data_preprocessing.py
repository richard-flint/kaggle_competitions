# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 10:04:11 2020

@author: richa
"""

#************************#
#*** Import libraries ***#
#************************#

import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import sklearn

def titatinc_data_preprocessing():
    
    #***************************#
    #*** User inputs ***#
    #***************************#
    
    #Select which features to include in the model (True=Yes, False=No):
    PassengerId=False
    Survived=False    #Note: This is always False as not a feature
    Pclass=True
    Name=False
    Sex=True
    Age=True        #Note: This feature has missing values
    SibSp=True
    Parch=True
    Ticket=False
    Fare=True
    Cabin=False     #Note: This feature has missing values
    Embarked=True
    
    #Set how to determine missing values in the dataset
    #Options include: remove_columns,remove_rows
    missing_values="remove_columns"
    
    #Feature scaling on or off
    feature_scaling_on_off=1 # 0 = off, 1 = normalisation, 2 = standardisation
    
    #***************************#
    #*** Read and clean data ***#
    #***************************#
    
    #Read data
    data = pd.read_csv("train.csv")
    
    #Plot to see if there is any missing data
    #plt.figure(1)
    #msno.matrix(data)
    
    #Manage missing values
    data,Age,Cabin=manage_missing_values(data,missing_values,Age,Cabin)
    
    #Shuffle data (for dividing into training and test datasets)
    data=data.sample(frac=1)
    
    #Clean data (e.g. convert categorical (text) to numerical data)
    data=clean_data(data)
    
    #Standardisation feature scaling
    #Note: It is easier to do this on a data frame and before removing columns
    if feature_scaling_on_off==2:
        data=feature_standardisation(data)
        
    #Remove columns that were not selected as feature vectors
    features_in,features_out=features_in_out(PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked)
    data_in=data.drop(columns=features_out)
    
    #Convert data to numpy array
    data_np=data_in.to_numpy()
    
    #Normalisation feature scaling
    #Note: It is easier to do this on a numpy matrix
    if feature_scaling_on_off==1:
        norm = sklearn.preprocessing.MinMaxScaler().fit(data_np)
        data_np=norm.transform(data_np)
    
    
    #**********************#
    #*** Define vectors ***#
    #**********************#
    
    #Define feature vector X
    X=data_np
    
    #Define output vector y
    y=data['Survived'].to_numpy()
    
    #Extract data parameters
    m=data_np.shape[0]
    n=data_np.shape[1]
    
    #Divide data into two datasets (one for training, one for testing)
    X_train,X_test,y_train,y_test=divide_data(X,y)
    
    #Extract data parameters for training and testing datasets
    m_train=X_train.shape[0]       #m = number of rows
    m_test=X_test.shape[0]
        
    return X,y,X_train,X_test,y_train,y_test,m,n,m_train,m_test

def features_in_out(PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked):
    
    #Create boolean with one hot encoded features
    Male=Sex
    Female=Sex
    Pclass_1=Pclass
    Pclass_2=Pclass
    Pclass_3=Pclass
    Embarked_S=Embarked
    Embarked_C=Embarked
    Embarked_Q=Embarked
    
    #Create array with boolean logic for each feature
    true_false_array=np.bool_([PassengerId,Survived,Pclass_1,Pclass_2,Pclass_3,Name,Male,Female,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked_S,Embarked_C,Embarked_Q])
    
    #Create array with feature names
    list_of_features=np.array(['PassengerId','Survived','Pclass_1','Pclass_2','Pclass_3','Name','Male','Female','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked_S','Embarked_C','Embarked_Q'])
    
    #Separate into "in and "out" arrays with feature names
    #(Note: This is used to select columns from the dataframe)
    features_in=list_of_features[true_false_array]
    features_out=list_of_features[~true_false_array]
    
    return features_in,features_out

def manage_missing_values(data,missing_values,Age,Cabin):
    if missing_values == "remove_columns":
        Age=False
        Cabin=False
    return data,Age,Cabin
    
def divide_data(X,y):
    number_of_rows=X.shape[0]
    midpoint=round(number_of_rows/2)
    X_train=X[0:midpoint,:]       #Take first half data for training
    X_test=X[(midpoint+1):,:]     #Take second half data for testing
    y_train=y[0:midpoint]
    y_test=y[(midpoint+1):]
    return X_train,X_test,y_train,y_test

def feature_standardisation(data):
    
    #Make copy of feature vector
    data_copy=data.copy()
    
    #Drop features that cannot be standardised
    data_copy=data_copy.drop(columns=['PassengerId','Survived','Pclass_1','Pclass_2','Pclass_3','Name','Male','Female','Ticket','Cabin','Embarked_S','Embarked_C','Embarked_Q'])
    data=data.drop(columns=['Age','SibSp','Parch','Fare'])
    
    #Normalise features
    data_copy=sklearn.preprocessing.normalize(data_copy)
    
    #Join back together again
    
    return data
    
def clean_data(data):
    
    #**********************#
    #*** Clean sex data ***#
    #**********************#
    
    #Convert categorical data into numerical data
    sex_str_to_num = {"Sex":     {"male": 1, "female": 0}}
    data=data.replace(sex_str_to_num)
    
    #One hot encoding
    male=data.Sex
    female=abs(male-1)
    data['Male']=male
    data['Female']=female
    
    #Remove sex as a feature
    data=data.drop(columns='Sex')
    
    #*************************#
    #*** Clean pclass data ***#
    #*************************#
    
    pclass=data.Pclass
    
    #One hot encoding
    pclass_1=np.where(pclass==1,1,0)
    pclass_2=np.where(pclass==2,1,0)
    pclass_3=np.where(pclass==3,1,0)
    
    #Add to dataframe
    data['Pclass_1']=pclass_1
    data['Pclass_2']=pclass_2
    data['Pclass_3']=pclass_3
    
    #Remove Pclass as a feature
    data=data.drop(columns='Pclass')
    
    #***************************#
    #*** Clean Embarked data ***#
    #***************************#
    
    embarked=data.Embarked
    
    #One hot encoding
    embarked_s=np.where(embarked=='S',1,0)
    embarked_c=np.where(embarked=='C',1,0)
    embarked_q=np.where(embarked=='Q',1,0)
    
    #Add to dataframe
    data['Embarked_S']=embarked_s
    data['Embarked_C']=embarked_c
    data['Embarked_Q']=embarked_q
    
    #Remove Pclass as a feature
    data=data.drop(columns='Embarked')
    
    return(data)