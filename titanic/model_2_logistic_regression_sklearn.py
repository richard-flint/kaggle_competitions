# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 12:12:24 2020

@author: richa
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

def logistic_regression_sklearn(X,y,X_train,X_test,y_train,y_test,m,n,m_train,m_test,plot_on):  
    
    #*******************************#
    #*** Define model parameters ***#
    #*******************************#
    
    regularisation = 1              #0 = off, 1 = on
    k=1
    alpha=0.001
    number_of_iterations=100000
    
    #***********************************************#
    #*** Initialise vectors for gradient descent ***#
    #***********************************************#
    
    clf=LogisticRegression().fit(X_train,y_train,sample_weight=None)
    y_predicted=clf.predict(X_test)
    h_test=clf.predict_proba(X_test)[:,1]
    
    return y_predicted,h_test