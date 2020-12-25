# -*- coding: utf-8 -*-
"""

"""

#************************#
#*** Import libraries ***#
#************************#

import numpy as np
import matplotlib.pyplot as plt

#****************************************************************************#
#******************************* Main function ******************************#
#****************************************************************************#

def logistic_regression_manual(X,y,X_train,X_test,y_train,y_test,m,n,m_train,m_test,plot_on):  
    
    #*******************************#
    #*** Define model parameters ***#
    #*******************************#
    
    regularisation = 1              #0 = off, 1 = on
    k=1
    alpha=0.001
    number_of_iterations=100000
    plot_on=1
    
    #*****************************************#
    #*** Add bias terms to feature vectors ***#
    #*****************************************#
    #Note: We dont do this in the data preprocessing, as sklearn automatically
    #      adds bias terms.
    X,X_train,X_test,n=add_bias_feature(X,y,X_train,X_test,y_train,y_test,m,n,m_train,m_test)
    
    #***********************************************#
    #*** Initialise vectors for gradient descent ***#
    #***********************************************#
    
    #Initialise theta
    theta=np.zeros(n,dtype='float64')       #initialise theta vector with 0s
    
    #Calculate initial hypothesis
    h=calculate_h(X_train,theta)
    
    #Calculate initial cost function
    if regularisation == 0:
        J=calculate_J_without_regularisation(h,y_train,m_train)
    elif regularisation == 1:
        J=calculate_J_with_regularisation(h,y_train,m_train,k,theta)
    
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
        theta=update_theta(X_train,y_train,theta,alpha,m_train)
        h=calculate_h(X_train,theta)
        if regularisation == 0:
            J=calculate_J_without_regularisation(h,y_train,m_train)
        elif regularisation == 1:
            J=calculate_J_with_regularisation(h,y_train,m_train,k,theta)
        J_history[i+1]=J
        theta_history[i+1,:]=theta
    
    #Plot the results
    if plot_on==1:
        plt.figure(2)
        plt.plot(J_history)
    
    #*************************************#
    #*** Predict survival on test data ***#
    #*************************************#
    
    #Cacluate h on test data
    h_test=calculate_h(X_test,theta)
    
    #Predict y values for test data
    y_predicted=np.where(h_test>=0.5,1,0)
    
    return h_test,y_predicted


#****************************************************************************#
#******************************* Sub-functions ******************************#
#****************************************************************************#

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

def add_bias_feature(X,y,X_train,X_test,y_train,y_test,m,n,m_train,m_test):
    
    #Add bias term to X
    x0=np.ones([m,1])
    X=np.hstack([x0,X])
    
    #Add bias term to X_train
    x0=np.ones([m_train,1])
    X_train=np.hstack([x0,X_train])
    
    #Add bias term to X_test
    x0=np.ones([m_test,1])
    X_test=np.hstack([x0,X_test])
    
    #Add one extra feature to n
    n=X.shape[1]
    
    return X,X_train,X_test,n

#****************************************************************************#