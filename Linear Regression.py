#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# To predict total sales using features like money spent on marketing individual items.

# In[1]:


# import basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Load the data
data=pd.read_csv('Advertising.csv')


# # Basic checks

# In[3]:


# find number of rows and columns
data.shape


# In[4]:


# find datatype
data.dtypes


# In[5]:


# find info
data.info()


# In[6]:


# print all the columns
data.columns


# In[7]:


# do statistical analysis
data.describe()


# # Exploratory data analysis

# ### univariate analysis
# Analysing single variable

# In[14]:


# Analyse TV

sns.distplot(x=data.TV,kde=True)
plt.show()


# In[15]:


# Analyse Radio
sns.distplot(x=data.Radio,kde=True)


# In[16]:


# Analyse Newspaper
sns.distplot(x=data.Newspaper,kde=True)


# In[17]:


# Analyse sales
sns.distplot(x=data.Sales,kde=True)


# In[ ]:


# Sales represents normal distribution.


# ### Bivariate analysis
# Analysing two variable

# In[22]:


# Analysing TV and sales
sns.relplot(x='TV',y='Sales',data=data)
plt.show()


# In[ ]:


# As we increase the amount of money spent on Television , Sales will increase.
# There is a positive relationship between sales and TV.


# In[23]:


# Anlysing Radio  and Sales
sns.relplot(x='Radio',y='Sales',data=data)
plt.show()


# In[ ]:


# There is no much trend or strong relationship between Radio and sales.
# Investing on Radio has  less impact on sales.


# In[21]:


# Analysing Newspaper and Sales
sns.relplot(x='Newspaper',y='Sales',data=data)


# In[ ]:


# There is no relationship between sales and Radio.
# Investing more or less on Newspaper advertisement will yield very less sales.

# Insights
* Investing high amount on TV marketing will yield more Sales compared to Radio and Newspaper.
* Investing on Radio will yield good sales compared to investing in Newspaper.
* Investing in Newspaper will result in less sales. So its good to avoid investing in Nespaper adverstising.
# In[24]:


# Multivariate analysis
sns.pairplot(data)


# # Data Preprocessing

# In[25]:


# check for missing values and handle them
data.isnull().sum()     # no missing values


# In[27]:


# check for duplicates
data.duplicated().sum()


# In[ ]:


# check for outliers using boxplot
# we use boxplot to check the direction of outliers
# We check outliers for continuous numerical columns


# In[28]:


# TV
sns.boxplot(data=data,x='TV')


# In[29]:


# Radio
sns.boxplot(data=data,x='Radio')


# In[30]:


# Newspaper
sns.boxplot(data=data,x='Newspaper')


# In[31]:


# Sales
sns.boxplot(data=data,x='Sales')


# In[ ]:


# Skipping  Scaling
# Skipping coverting categorical data into numerical


# # Feature Engineering
# 
# 

# In[35]:


#Drop the irrelevant columns
data.drop('Unnamed: 0',axis=1,inplace=True)

# heat map
* To find variables which has high correlation with respect to Target.
* If correlation of variable is high with target we include them and if correlation is less with target we drop them.
# In[37]:


data.corr()


# In[39]:


sns.heatmap(data.corr(),annot=True)

# Tv and Radio has high correlation with target, so we include them in data.
# Newspaper has very less correlation with sales so we can drop that column but we are not dropping bcoz data is small.
# In[41]:


# Check for multicollinearity: Find correlation among input variable
# If any two input variable represents high correlation then we have to drop one of the column.
sns.heatmap(data.drop('Sales',axis=1).corr(),annot =True)


# In[ ]:


# Clearly there is no high correlation among input variables . So we include all the columns


# # split data into x and y
# 

# In[42]:


x=data.drop('Sales',axis=1)
y=data.Sales


# In[50]:


# Split data for training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)


# In[51]:


x_train.shape


# In[52]:


x_test.shape


# In[53]:


y_train.shape


# In[54]:


y_test.shape

# Sklearn is one of the open source scientific library that has many packages or modules related to data preprocessing and Machine learning.
# random state controls randomness while shuffling data
# # Apply ML Model

# # Linear Regression

# In[57]:


# Import Linear Regression
from sklearn.linear_model import LinearRegression
# Initialise the model
model=LinearRegression()
# Train a model with x_train and y_train using fit
model.fit(x_train,y_train)
#  Predict values using x_test
y_pred=model.predict(x_test)


# In[58]:


y_pred


# In[60]:


# coefficient
model.coef_


# In[61]:


# intercept
model.intercept_


# In[59]:


y_test


# # Evaluate the model

# In[ ]:


# MSE
# MAE
# RMSE
# R2-score
# adjusted R2-score
# To evaluate model


# In[62]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# In[63]:


# find mse
mse=mean_squared_error(y_test,y_pred)
mse


# In[65]:


# find mae
mae=mean_absolute_error(y_test,y_pred)
mae


# In[66]:


# find RMSE
rmse=np.sqrt(mse)
rmse


# In[67]:


# find r2_score
r2_score=r2_score(y_test,y_pred)
r2_score


# In[ ]:


# 90% better model
# Model has learnt 90% of the information/data.


# In[68]:


y_test.shape


# In[69]:


# adjusted r2_score
adj_r2_score=1-(1-0.90)*(40-1)/(40-3-1)
adj_r2_score


# In[ ]:


# adjusted r2 score should always be less than r2 score in order to say better model.


# # predictions

# In[76]:


# what will be the sales if amount spent on TV=1000, Radio=50 , Newspaper=20
model.predict([[1000,10,20]])


# In[77]:


model.predict([[10,500,20]])


# In[75]:


model.predict([[10,10,1000]])


# In[80]:


model.predict([[200,700,0]])


# # Conclusion
# * Linear regression is 90% better model to make predictions in the future.
# * Investing more on TV and Radio will return good sales compared to Newspaper.

# In[ ]:


# train a model: x_train ,y_train
# prediction(y_pred):x_test
# evaluate:y_test,y_pred


# In[ ]:





# In[ ]:





# In[ ]:




