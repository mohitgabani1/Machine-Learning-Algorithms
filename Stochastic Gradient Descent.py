#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from datetime import *


# In[2]:


X, y = load_diabetes(return_X_y = True)


# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)


# In[4]:


X_train.shape


# #### Linear Regression: 

# In[5]:


from sklearn.linear_model import LinearRegression


# In[6]:


lr = LinearRegression()


# In[7]:


lr.fit(X_train, y_train)


# In[8]:


y_pred_lr = lr.predict(X_test)


# In[9]:


lr.coef_


# In[10]:


lr.intercept_


# In[11]:


from sklearn.metrics import r2_score
r2_score(y_test, y_pred_lr)


# #### Stochastic Gradient Descent Regressor: 

# In[12]:


from sklearn.linear_model import SGDRegressor


# In[13]:


sgd = SGDRegressor(max_iter = 1000, learning_rate = 'constant',eta0=0.1)


# In[14]:


sgd.fit(X_train, y_train)


# In[15]:


y_pred = sgd.predict(X_test)


# In[16]:


y_pred.shape


# In[17]:


sgd.coef_


# In[18]:


sgd.intercept_


# In[19]:


print('r2 score:', r2_score(y_test,y_pred))


# #### Stochastic Gradient Descent Class: 

# In[20]:


class SGDregressor:
    
    def __init__(self,learning_rate, epochs):
        
        self.lr = learning_rate
        self.epochs = epochs
        self.intercept = None
        self.coeff = None
        
    def fit(self, x_train, y_train):
        
        # defining initial value of intercept and coefficients
        self.intercept = 0
        self.coeff = np.ones(x_train.shape[1])
        
        for i in range(self.epochs):
            
            for j in range(x_train.shape[0]):

                # generating random number for picking up random row
                idx = np.random.randint(0, x_train.shape[0])
                
                # updating the value of intercept
                y_hat = np.dot(x_train[idx],self.coeff) + self.intercept
                intercept_der = -2*np.sum(y_train[idx] - y_hat)
                self.intercept = self.intercept - self.lr*intercept_der

                # updating the value of coefficients:
                coeff_der = -2*np.dot(np.transpose(y_train[idx] - y_hat),x_train[idx])
                self.coeff = self.coeff - self.lr*coeff_der

        print(self.intercept, self.coeff)
        
    def predict(self, x_test):
        return np.dot(x_test, self.coeff) + self.intercept
    


# In[21]:


sgd_class = SGDregressor(0.01,100)


# In[22]:


start = datetime.now()
sgd_class.fit(X_train, y_train)
print('time taken', datetime.now() - start)


# In[23]:


y_pred_sgd_class = sgd_class.predict(X_test) 


# In[24]:


r2_score(y_test, y_pred_sgd_class)

