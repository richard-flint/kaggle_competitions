# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 21:32:39 2020

@author: richa
"""

import numpy as np

def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g

    
def calculate_h(X, theta):
    z=np.dot(X,theta)
    h=sigmoid(z)
    return h
    
def calculate_J(h,y,m):
    J=(1/m)*(np.dot(-y.T,np.log(h))-np.dot((1-y).T,np.log(1-h)))
    return J

def update_theta(X,y,theta,alpha,m):
    z=np.dot(X,theta)
    g=sigmoid(z)
    theta=theta-(alpha/m)*np.dot(X.T,(g-y))
    return theta

def clean_data(data):
    
    #Clean sex data
    sex_str_to_num = {"Sex":     {"male": 1, "female": 0}}
    data=data.replace(sex_str_to_num)
    
    
    return(data)
    
def divide_data(X,y):
    number_of_rows=X.shape[0]
    midpoint=round(number_of_rows/2)
    X_train=X[0:midpoint,:]       #Take first half data for training
    X_test=X[(midpoint+1):,:]     #Take second half data for testing
    y_train=y[0:midpoint]
    y_test=y[(midpoint+1):]
    return X_train,X_test,y_train,y_test

def define_feature_vector(data_np,m):
    
    #Remove y column
    X=np.delete(data_np,1,axis=1)
    
    #Add x0 column
    x0=np.ones(m)
    X=np.column_stack([x0,X])
    
    #Count number of features
    n=X.shape[1]
    
    return X,n