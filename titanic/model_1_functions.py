# -*- coding: utf-8 -*-
"""
Summary: This script contains the user defined functions for the main script.

"""

import numpy as np

def manage_missing_values(data,missing_values,Age,Cabin):
    if missing_values == "remove_columns":
        Age=False
        Cabin=False
    return data,Age,Cabin

def features_in_out(PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked):
    true_false_array=np.bool_([PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked])
    list_of_features=np.array(['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])
    features_in=list_of_features[true_false_array]
    features_out=list_of_features[~true_false_array]
    return features_in,features_out

def sigmoid(z):
    g=1/(1+np.exp(-z))
    return g

def calculate_h(X, theta):
    z=np.dot(X,theta)
    h=sigmoid(z)
    return h
    
def calculate_J_without_regularisation(h,y,m):
    J=(1/m)*(np.dot(-y.T,np.log(h))-np.dot((1-y).T,np.log(1-h)))
    return J

def calculate_J_with_regularisation(h,y,m,k,theta):
    J=calculate_J_without_regularisation(h,y,m)
    J=J+((k/(2*m))*np.dot(theta.T,theta))
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

#*******************************************#
#*** Functions for measuring performance ***#
#*******************************************#

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

