# -*- coding: utf-8 -*-
"""
Summary: This is a script that attempts to predict survivors from Titanic.
This is the introductory Kaggle competition.
See https://www.kaggle.com/c/titanic

Note: This is currently a work in progress (WIP) and is incomplete.

To do:
    * Implement a random forest
    * Implement a neural network
    * Implement a SVM
    * Improve approach to missing values
    * Do some data engineering on features like cabin no.
    * Implement learner curves
    * Add feature scaling/normalisation
"""

#***********************#
#*** Clear workspace ***#
#***********************#

from IPython import get_ipython
get_ipython().magic('reset -sf')

#************************#
#*** Import libraries ***#
#************************#

import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import sklearn

#Import user defined functions
from data_preprocessing import titatinc_data_preprocessing
from model_1_logistic_regression_manual import logistic_regression_manual
from model_evaluation_manual import model_evaluation_manual
from model_2_logistic_regression_sklearn import logistic_regression_sklearn
from model_evaluation_sklearn import model_evaluation_sklearn

#****************************************************************************#
#***************************** Data preprocessing ***************************#
#****************************************************************************#

X,y,X_train,X_test,y_train,y_test,m,n,m_train,m_test = titatinc_data_preprocessing()

#****************************************************************************#
#******************************* Implement models ***************************#
#****************************************************************************#

#Select which models to implement
logistic_regression_manual_on_off=1
logistic_regression_sklearn_on_off=0

#Select whether to display plots
plot_on=1       #0=off,1=on

#Logistic regression (manual) using gradient descent
if logistic_regression_manual_on_off==1:
    h_test,y_predicted=logistic_regression_manual(X,y,X_train,X_test,y_train,y_test,m,n,m_train,m_test,plot_on)

#Logistic regression (sklearn)
if logistic_regression_sklearn_on_off==1:
    y_predicted,h_test=logistic_regression_sklearn(X,y,X_train,X_test,y_train,y_test,m,n,m_train,m_test,plot_on)


#****************************************************************************#
#*************************** Evaluate model performance *********************#
#****************************************************************************#

#Select which models to implement
manual_evaluation_on_off=1
sklearn_evaluation_on_off=0

#Other options
print_results=1
plot_on=1

#Implement manual evaluation
if manual_evaluation_on_off==1:
    accuracy,recall,precision,specificity,f_measure,TPR,FPR,ROC_TPR,ROC_FPR=model_evaluation_manual(y_predicted,y_test,h_test,print_results,m_test,plot_on)

#Implement sklearn evaluation
if sklearn_evaluation_on_off==1:
    confusion_matrix,accuracy,precision,recall,f_measure,roc_curve,roc_auc=model_evaluation_sklearn(y_test,y_predicted,h_test,print_results,plot_on)
    
#****************************************************************************#

