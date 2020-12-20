# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#************************#
#*** Import libraries ***#
#************************#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import model_1_functions as udf

#*******************#
#*** User inputs ***#
#*******************#

#Select which features to include in the model (True=Yes, False=No):
PassengerId=False
Survived=True     #Note: This is always True
Pclass=True
Name=False
Sex=True
Age=True
SibSp=True
Parch=True
Ticket=False
Fare=True
Cabin=False
Embarked=False

#Save this information in an array
features_in_out=np.array([PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked])

#Gradient descent parameters
alpha=3
number_of_iterations=100000

#***************************#
#*** Read and clean data ***#
#***************************#

#Read data
data = pd.read_csv("train.csv")

#Shuffle data
data=data.sample(frac=1)

#Clean data
data=udf.clean_data(data)

#Convert to numpy array
data_np=data.to_numpy()

#Remove columns that were not selected as feature vectors
data_np=data_np[:,features_in_out]

#Remove rows with blank (nan) values
data_np=data_np[~np.isnan(data_np).any(axis=1)]

#Extract data parameters
m=data_np.shape[0]

#Define feature vector X (and number of features n)
X,n=udf.define_feature_vector(data_np,m)

#Define output vector y
y=data_np[:,1]

#Divide data into two datasets (one for training, one for test)
X_train,X_test,y_train,y_test=udf.divide_data(X,y)

#Extract data parameters
m_train=X_train.shape[0]       #m = number of rows
m_test=X_test.shape[0]

#**************************#
#*** Initialise vectors ***#
#**************************#

#Initialise theta
theta=np.zeros(n,dtype='float64')       #initialise theta at 0

#Calculate initial hypothesis
h=udf.calculate_h(X_train,theta)

#Calculate initial cost function
J=udf.calculate_J(h,y_train,m_train)

#Initialise history vectors
J_history=np.zeros(number_of_iterations+1)
theta_history=np.zeros([number_of_iterations+1,n])

#Save initial values
theta_history[0,:]=theta
J_history[0]=J

#******************************************************#
#*** Use gradient descent to minimise cost function ***#
#******************************************************#

#Iterate through gradient descent
for i in range(number_of_iterations):
    theta=udf.update_theta(X_train,y_train,theta,alpha,m_train)
    h=udf.calculate_h(X_train,theta)
    J=udf.calculate_J(h,y_train,m_train)
    J_history[i+1]=J
    theta_history[i+1,:]=theta

#Plot the results
plt.plot(J_history)
#plt.scatter(theta_history[:,0],theta_history[:,1])

#*************************#
#*** Test on test data ***#
#*************************#

#Cacluate h on test data
h_test=udf.calculate_h(X_test,theta)

#Predict y values for test data
y_predicted=np.where(h_test>=0.5,1,0)

#Compare predicted against actual
correct_vs_incorrect=np.where(y_predicted==y_test,1,0)

#Calculate percentage correct
percentage_correct=sum(correct_vs_incorrect)/m_test

#Print results
print(percentage_correct)

#*********************#
#*** sci-kit learn ***#
#*********************#
