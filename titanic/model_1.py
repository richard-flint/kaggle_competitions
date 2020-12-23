# -*- coding: utf-8 -*-
"""
Summary: This is a script that attempts to predict survivors from Titanic.
This is the introductory Kaggle competition.
See https://www.kaggle.com/c/titanic

Note: This is currently a work in progress (WIP) and is incomplete.

To do:
    * Implement using skikit-learn
    * Implement a random forest
    * Implement a neural network
    * Improve/trial different approaches to missing values
    * Do some data engineering on features like cabin no.
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
import model_1_functions as udf

#*******************#
#*** User inputs ***#
#*******************#

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
Embarked=False

#Set gradient descent parameters
regularisation = 1 #0 = off, 1 = on
k=1
alpha=0.001
number_of_iterations=100000

#Set how to determine missing values in the dataset
#Options include: remove_columns,remove_rows
missing_values="remove_columns"


#***************************#
#*** Read and clean data ***#
#***************************#

#Read data
data = pd.read_csv("train.csv")

#Plot to see if there is any missing data
#plt.figure(1)
#msno.matrix(data)

#Manage missing values
data,Age,Cabin=udf.manage_missing_values(data,missing_values,Age,Cabin)

#Shuffle data (for dividing into training and test datasets)
data=data.sample(frac=1)

#Clean data (e.g. convert categorical (text) to numerical data)
data=udf.clean_data(data)

#Remove columns that were not selected as feature vectors
features_in,features_out=udf.features_in_out(PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked)
data_in=data.drop(columns=features_out)

#Convert data to numpy array
data_np=data_in.to_numpy()

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
X_train,X_test,y_train,y_test=udf.divide_data(X,y)

#Extract data parameters for training and testing datasets
m_train=X_train.shape[0]       #m = number of rows
m_test=X_test.shape[0]

#***********************************************#
#*** Initialise vectors for gradient descent ***#
#***********************************************#

#Initialise theta
theta=np.zeros(n,dtype='float64')       #initialise theta at 0

#Calculate initial hypothesis
h=udf.calculate_h(X_train,theta)

#Calculate initial cost function
if regularisation == 0:
    J=udf.calculate_J_without_regularisation(h,y_train,m_train)
elif regularisation == 1:
    J=udf.calculate_J_with_regularisation(h,y_train,m_train,k,theta)

#Initialise history vectors
J_history=np.zeros(number_of_iterations+1)
theta_history=np.zeros([number_of_iterations+1,n])

#Save initial values
theta_history[0,:]=theta
J_history[0]=J

#**********************************#
#*** Implement gradient descent ***#
#**********************************#

#Iterate through gradient descent
for i in range(number_of_iterations):
    theta=udf.update_theta(X_train,y_train,theta,alpha,m_train)
    h=udf.calculate_h(X_train,theta)
    if regularisation == 0:
        J=udf.calculate_J_without_regularisation(h,y_train,m_train)
    elif regularisation == 1:
        J=udf.calculate_J_with_regularisation(h,y_train,m_train,k,theta)
    J_history[i+1]=J
    theta_history[i+1,:]=theta

#Plot the results
plt.figure(2)
plt.plot(J_history)
#plt.scatter(theta_history[:,0],theta_history[:,1])

#*************************************#
#*** Predict survival on test data ***#
#*************************************#

#Cacluate h on test data
h_test=udf.calculate_h(X_test,theta)

#Predict y values for test data
y_predicted=np.where(h_test>=0.5,1,0)

#**************************#
#*** Review performance ***#
#**************************#

#Calculate confusion matrix
true_positive_count,true_negative_count,false_positive_count,false_negative_count=udf.calculate_confustion_matrix(y_predicted,y_test)

#Calculate performance metrics
accuracy,recall,precision,specificity,f_measure,TPR,FPR=udf.calculate_performance_metrics(true_positive_count,true_negative_count,false_positive_count,false_negative_count,m_test)

#Print performance metrics
print("Accuracy = ",round(accuracy,2),"%")
print("Recall = ",round(recall,2),"%")
print("Precision = ",round(precision,2),"%")
print("Specificity = ",round(specificity,2),"%")
print("F-measure = ",round(f_measure,2),"%")
print("TPR = ",round(TPR,2),"%")
print("FPR = ",round(FPR,2),"%")

#Draw ROC and calculate ROC AUC
number_of_roc_steps=21
ROC_TPR=np.zeros(number_of_roc_steps,dtype='float64')
ROC_FPR=np.zeros(number_of_roc_steps,dtype='float64')
roc_increment=1/(number_of_roc_steps-1)
threshold=0

for i in range(number_of_roc_steps):
    #Estimate predicted values using variable threshold
    y_predicted_roc=np.where(h_test>=threshold,1,0)
    
    #Calculate performance
    roc_true_positive_count,roc_true_negative_count,roc_false_positive_count,roc_false_negative_count=udf.calculate_confustion_matrix(y_predicted_roc,y_test)
    roc_accuracy,roc_recall,roc_precision,roc_specificity,roc_f_measure,roc_TPR,roc_FPR=udf.calculate_performance_metrics(roc_true_positive_count,roc_true_negative_count,roc_false_positive_count,roc_false_negative_count,m_test)
    
    #Record values in vectors
    ROC_TPR[i]=roc_TPR
    ROC_FPR[i]=roc_FPR
    
    #Increment threshold value
    threshold=threshold+roc_increment
    
#Draw ROC graph
plt.figure(3)
plt.step(ROC_FPR,ROC_TPR)
plt.plot(np.array([0,1]),np.array([0,1]))

#*********************#
#*** sci-kit learn ***#
#*********************#
