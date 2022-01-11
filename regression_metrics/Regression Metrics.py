#!/usr/bin/env python
# coding: utf-8

# ### Boston Housing Data
# 
# In order to gain a better understanding of the metrics used in regression settings, we will be looking at the Boston Housing dataset.  
# 
# First use the cell below to read in the dataset and set up the training and testing data that will be used for the rest of this problem.

# In[18]:


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
import tests2 as t

boston = load_boston()
y = boston.target
X = boston.data

X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42)

print(len(boston))
print(len(y), 'Y')
print(len(X), 'X')
print(len(X_train), 'X_train')
print(len(X_test), 'X_test')
print(len(y_train), 'Y_train')
print(len(y_test), 'Y_test')


# > **Step 1:** Before we get too far, let's do a quick check of the models that you can use in this situation given that you are working on a regression problem.  Use the dictionary and corresponding letters below to provide all the possible models you might choose to use.

# In[4]:


# When can you use the model - use each option as many times as necessary
a = 'regression'
b = 'classification'
c = 'both regression and classification'

models = {
    'decision trees': c,
    'random forest': c,
    'adaptive boosting': c,
    'logistic regression': b,
    'linear regression': a
}

#checks your answer, no need to change this code
t.q1_check(models)


# > **Step 2:** Now for each of the models you found in the previous question that can be used for regression problems, import them using sklearn.

# In[5]:


# Import models from sklearn - notice you will want to use 
# the regressor version (not classifier) - googling to find 
# each of these is what we all do!
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression


# > **Step 3:** Now that you have imported the 4 models that can be used for regression problems, instantate each below.

# In[6]:


# Instantiate each of the models you imported
# For now use the defaults for all the hyperparameters
rf_rgr_model = RandomForestRegressor()
dt_rgr_model = DecisionTreeRegressor()
ada_boots_rgr_model = AdaBoostRegressor()
linear_model = LinearRegression()


# > **Step 4:** Fit each of your instantiated models on the training data.

# In[7]:


# Fit each of your models using the training data
rf_rgr_model.fit(X_train, y_train)
dt_rgr_model.fit(X_train, y_train)
ada_boots_rgr_model.fit(X_train, y_train)
linear_model.fit(X_train, y_train)


# > **Step 5:** Use each of your models to predict on the test data.

# In[8]:


# Predict on the test values for each model
rf_pred = rf_rgr_model.predict(X_test)
dt_pred = dt_rgr_model.predict(X_test)
ada_pred = ada_boots_rgr_model.predict(X_test)
linear_pred = linear_model.predict(X_test)


# > **Step 6:** Now for the information related to this lesson.  Use the dictionary to match the metrics that are used for regression and those that are for classification.

# In[11]:


# potential model options
a = 'regression'
b = 'classification'
c = 'both regression and classification'

#
metrics = {
    'precision': b,
    'recall': b,
    'accuracy': b,
    'r2_score': a,
    'mean_squared_error': a,
    'area_under_curve': b, 
    'mean_absolute_area': a 
}

#checks your answer, no need to change this code
t.q6_check(metrics)


# > **Step 6:** Now that you have identified the metrics that can be used in for regression problems, use sklearn to import them.

# In[13]:


# Import the metrics from sklearn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# > **Step 7:** Similar to what you did with classification models, let's make sure you are comfortable with how exactly each of these metrics is being calculated.  We can then match the value to what sklearn provides.

# In[14]:


def r2(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the r-squared score as a float
    '''
    sse = np.sum((actual-preds)**2)
    sst = np.sum((actual-np.mean(actual))**2)
    return 1 - sse/sst

# Check solution matches sklearn
print(r2(y_test, rf_pred), 'Random Forest Regressor')
print(r2_score(y_test, rf_pred))
print("Since the above match, we can see that we have correctly calculated the r2 value.")

print(r2(y_test, dt_pred), 'Decision Tree Regressor')
print(r2_score(y_test, dt_pred))
print("Since the above match, we can see that we have correctly calculated the r2 value.")

print(r2(y_test, ada_pred), 'ADA Boost Regressor')
print(r2_score(y_test, ada_pred))
print("Since the above match, we can see that we have correctly calculated the r2 value.")

print(r2(y_test, linear_pred), 'Linear Regression')
print(r2_score(y_test, linear_pred))
print("Since the above match, we can see that we have correctly calculated the r2 value.")


# > **Step 8:** Your turn fill in the function below and see if your result matches the built in for mean_squared_error. 

# In[34]:


def mse(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the mean squared error as a float
    '''
    
    return np.sum((actual-preds)**2) / len(actual) # calculate mse here


# Check your solution matches sklearn

print(mse(y_test, rf_pred), 'Random Forest Regressor')
print(mean_squared_error(y_test, rf_pred))
print("Since the above match, we can see that we have correctly calculated the r2 value.")

print(mse(y_test, dt_pred), 'Decision Tree Regressor')
print(mean_squared_error(y_test, dt_pred))
print("Since the above match, we can see that we have correctly calculated the r2 value.")

print(mse(y_test, ada_pred), 'ADA Boost Regressor')
print(mean_squared_error(y_test, ada_pred))
print("Since the above match, we can see that we have correctly calculated the r2 value.")

print(mse(y_test, linear_pred), 'Linear Regression')
print(mean_squared_error(y_test, linear_pred))
print("Since the above match, we can see that we have correctly calculated the r2 value.")


# > **Step 9:** Now one last time - complete the function related to mean absolute error.  Then check your function against the sklearn metric to assure they match. 

# In[35]:


def mae(actual, preds):
    '''
    INPUT:
    actual - numpy array or pd series of actual y values
    preds - numpy array or pd series of predicted y values
    OUTPUT:
    returns the mean absolute error as a float
    '''
    
    return np.sum(np.abs(actual-preds)) / len(actual) # calculate the mae here

# Check your solution matches sklearn

print(mae(y_test, rf_pred), 'Random Forest Regressor')
print(mean_absolute_error(y_test, rf_pred))
print("Since the above match, we can see that we have correctly calculated the r2 value.")

print(mae(y_test, dt_pred), 'Decision Tree Regressor')
print(mean_absolute_error(y_test, dt_pred))
print("Since the above match, we can see that we have correctly calculated the r2 value.")

print(mae(y_test, ada_pred), 'ADA Boost Regressor')
print(mean_absolute_error(y_test, ada_pred))
print("Since the above match, we can see that we have correctly calculated the r2 value.")

print(mae(y_test, linear_pred), 'Linear Regression')
print(mean_absolute_error(y_test, linear_pred))
print("Since the above match, we can see that we have correctly calculated the r2 value.")


# > **Step 10:** Which model performed the best in terms of each of the metrics?  Note that r2 and mse will always match, but the mae may give a different best model.  Use the dictionary and space below to match the best model via each metric.

# In[38]:


#match each metric to the model that performed best on it
a = 'decision tree'
b = 'random forest'
c = 'adaptive boosting'
d = 'linear regression'


best_fit = {
    'mse': b,
    'r2': b,
    'mae': b
}

#Tests your answer - don't change this code
t.check_ten(best_fit)


# In[ ]:


# cells for work


# In[ ]:





# In[ ]:




