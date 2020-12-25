# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:25:38 2020

@author: richa
"""
import numpy as np
import matplotlib.pyplot as plt

#****************************************************************************#
#******************************* Main function ******************************#
#****************************************************************************#

def model_evaluation_manual(y_predicted,y_test,h_test,print_results,m_test,plot_on):

    #Calculate confusion matrix
    true_positive_count,true_negative_count,false_positive_count,false_negative_count=calculate_confustion_matrix(y_predicted,y_test)
    
    #Calculate performance metrics
    accuracy,recall,precision,specificity,f_measure,TPR,FPR=calculate_performance_metrics(true_positive_count,true_negative_count,false_positive_count,false_negative_count,m_test)
    
    #Print performance metrics
    if print_results==1:
        print("Accuracy = ",round(accuracy,2),"%")
        print("Recall = ",round(recall,2),"%")
        print("Precision = ",round(precision,2),"%")
        print("Specificity = ",round(specificity,2),"%")
        print("F-measure = ",round(f_measure,2),"%")
        print("TPR = ",round(TPR,2),"%")
        print("FPR = ",round(FPR,2),"%")
    
    #Draw ROC and calculate ROC AUC
    number_of_roc_steps=101
    ROC_TPR=np.zeros(number_of_roc_steps,dtype='float64')
    ROC_FPR=np.zeros(number_of_roc_steps,dtype='float64')
    roc_increment=1/(number_of_roc_steps-1)
    threshold=0
    
    for i in range(number_of_roc_steps):
        #Estimate predicted values using variable threshold
        y_predicted_roc=np.where(h_test>=threshold,1,0)
        
        #Calculate performance
        roc_true_positive_count,roc_true_negative_count,roc_false_positive_count,roc_false_negative_count=calculate_confustion_matrix(y_predicted_roc,y_test)
        roc_accuracy,roc_recall,roc_precision,roc_specificity,roc_f_measure,roc_TPR,roc_FPR=calculate_performance_metrics(roc_true_positive_count,roc_true_negative_count,roc_false_positive_count,roc_false_negative_count,m_test)
        
        #Record values in vectors
        ROC_TPR[i]=roc_TPR
        ROC_FPR[i]=roc_FPR
        
        #Increment threshold value
        threshold=threshold+roc_increment
        
    #Draw ROC graph
    if plot_on==1:
        plt.figure(3)
        plt.step(ROC_FPR,ROC_TPR)
        plt.plot(np.array([0,1]),np.array([0,1]))
        
    return accuracy,recall,precision,specificity,f_measure,TPR,FPR,ROC_TPR,ROC_FPR
    
#****************************************************************************#
#****************************** Sub-functions *******************************#
#****************************************************************************#

def identify_true_positives(y_predicted,y_test):
    true_positives=np.where(y_predicted==0,0,(np.where(y_predicted==y_test,1,0)))
    return true_positives

def identify_true_negatives(y_predicted,y_test):
    true_negatives=np.where(y_predicted==1,0,(np.where(y_predicted==y_test,1,0)))
    return true_negatives

def identify_false_positives(y_predicted,y_test):
    false_positives=np.where(y_predicted==0,0,(np.where(y_predicted!=y_test,1,0)))
    return false_positives
    
def identify_false_negatives(y_predicted,y_test):
    false_negatives=np.where(y_predicted==1,0,(np.where(y_predicted!=y_test,1,0)))
    return false_negatives

def calculate_confustion_matrix(y_predicted,y_test):
    #Identify TP,TN,FP and FN in data
    true_positives=identify_true_positives(y_predicted,y_test)
    true_negatives=identify_true_negatives(y_predicted,y_test)
    false_positives=identify_false_positives(y_predicted,y_test)
    false_negatives=identify_false_negatives(y_predicted,y_test)
    #Count
    true_positive_count=np.sum(true_positives)
    true_negative_count=np.sum(true_negatives)
    false_positive_count=np.sum(false_positives)
    false_negative_count=np.sum(false_negatives)
    return true_positive_count,true_negative_count,false_positive_count,false_negative_count

def check_confusion_matrix(y_predicted,y_test,true_positives,true_negatives,false_positives,false_negatives):
    
    #Create vector with TP,TN,FP and FN labels
    confusion_vector=np.where(true_positives==1,'TP',
                          np.where(true_negatives==1,'TN',
                          np.where(false_positives==1,'FP',
                          np.where(false_negatives==1,'FN','NA'))))

    #Create matrix that allows easy checking
    comparison_matrix=np.vstack([y_predicted,y_test,confusion_vector]).T
    
    #Return matrix
    return comparison_matrix

def calculate_performance_metrics(true_positive_count,true_negative_count,false_positive_count,false_negative_count,m_test):
    accuracy=calculate_accuracy(true_positive_count,true_negative_count,m_test)
    recall=calculate_recall(true_positive_count,false_negative_count)
    precision=calculate_precision(true_positive_count,false_positive_count)
    specificity=calculate_specificity(true_negative_count,false_positive_count)
    f_measure=calculate_f_measure(recall,precision)
    TPR=recall
    FPR=1-specificity
    return accuracy,recall,precision,specificity,f_measure,TPR,FPR

def calculate_accuracy(true_positive_count,true_negative_count,m_test):
    accuracy=(true_positive_count+true_negative_count)/m_test
    return accuracy

def calculate_recall(true_positive_count,false_negative_count):
    if true_positive_count==0 and false_negative_count==0:
        recall=0
    else:
        recall=true_positive_count/(true_positive_count+false_negative_count)
    return recall

def calculate_precision(true_positive_count,false_positive_count):
    if true_positive_count==0 and false_positive_count==0:
        precision=0
    else:
        precision=true_positive_count/(true_positive_count+false_positive_count)
    return precision

def calculate_specificity(true_negative_count,false_positive_count):
    if true_negative_count==0 and false_positive_count==0:
        specificity=0
    else:
        specificity=true_negative_count/(true_negative_count+false_positive_count)
    return specificity

def calculate_f_measure(recall,precision):
    if recall==0 and precision==0:
        f_measure=0
    else:
        f_measure=(2*recall*precision)/(recall+precision)
    return f_measure