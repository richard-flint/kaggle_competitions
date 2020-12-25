# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 15:36:05 2020

@author: richa
"""

import numpy as np
import sklearn
import matplotlib.pyplot as plt

def model_evaluation_sklearn(y_true,y_predicted,h_test,print_results,plot_on):
    
    #Confusion matrix
    confusion_matrix=sklearn.metrics.confusion_matrix(y_true,y_predicted)
    
    #Accuracy
    accuracy=sklearn.metrics.accuracy_score(y_true,y_predicted)
    
    #Precision
    precision=sklearn.metrics.precision_score(y_true,y_predicted)
    
    #Recall
    recall=sklearn.metrics.recall_score(y_true,y_predicted)
    
    #F-measure
    f_measure=sklearn.metrics.f1_score(y_true,y_predicted)
    
    #ROC
    roc_curve=sklearn.metrics.roc_curve(y_true,h_test)
    ROC_TPR=roc_curve[1]
    ROC_FPR=roc_curve[0]
    
    #ROC AUC
    roc_auc=sklearn.metrics.roc_auc_score(y_true,h_test)
    
    #Print metrics
    if print_results==1:
        print("Accuracy = ",round(accuracy,2),"%")
        print("Recall = ",round(recall,2),"%")
        print("Precision = ",round(precision,2),"%")
        print("F-measure = ",round(f_measure,2),"%")
        print("ROC AUC = ",round(roc_auc,2),"%")
        
    #Plot ROC curve
    if plot_on==1:
        plt.figure(5)
        plt.plot(ROC_FPR,ROC_TPR)
        plt.plot(np.array([0,1]),np.array([0,1]))
    
    return confusion_matrix,accuracy,precision,recall,f_measure,roc_curve,roc_auc