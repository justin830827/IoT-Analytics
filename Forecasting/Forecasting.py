#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.arima_model import ARIMA


# ## Data Preparation
# The data set comes from a local weather station, and it includes hourly observations for
# temperature, humidity, and atmospheric pressure. There are approximately six months’ data. For
# this project, you will only use the temperatures. Ignore the other two readings. All the class files
# are posted in Moodle. Use the one that has your ID.
# 
# Partition your data set into two roughly equal parts of contiguous data. The first part is the training
# set and the second part is testing set. Use the training set to do tasks 2, 3, and 4. Use the testing set
# to do task 5.

# In[2]:


# Read dataset and extract temp (1st. column).
def read_data():
    df = pd.read_csv('./data/200263453.csv')
    df = df.iloc[:,:1]
    df = df.rename(columns = {'2m Temperature (hr. avg) (F)': 'Temperature'})
    return df

data = read_data()
data.describe()


# ## Task 1. Check for stationarity
# Plot the entire time series (i.e. both training and testing sets) and check it visually for stationarity.
# If it is not stationary, make appropriate transformations as discussed in section 6.1.2 of the book.
# Comment on your conclusions.

# In[3]:


def plot_timeseries(data, title):
    mean = np.mean(data)
    plt.figure(figsize=(15,6))
    plt.plot(data)
    plt.axhline(y=mean,linewidth=4, color='r')
    plt.xlabel('Time')
    plt.ylabel('Temperature (F)')
    plt.title(title)
    plt.savefig('{}.png'.format(title))
    
plot_timeseries(data['Temperature'], 'Temp. Time Series plot')


# In[4]:


data['diff'] = data['Temperature'] - data['Temperature'].shift(1)
plot_timeseries(data['diff'], 'Temp. Time Series plot (diff1)')


# In[5]:


data['diff2'] = data['diff'] - data['diff'].shift(1)
plot_timeseries(data['diff2'], 'Temp. Time Series plot (diff2)')


# In[6]:


data['log'] = np.log(data['Temperature'])
plot_timeseries(data['log'], 'Temp. Time Series plot (log)')


# In[7]:


data['log_diff'] = data['log'] - data['log'].shift(1)
plot_timeseries(data['log_diff'], 'Temp. Time Series plot (log&diff)')


# In[8]:


data['Temperature'].hist()


# In[9]:


data['diff'].hist()


# In[10]:


data['diff2'].hist()


# In[11]:


data['log'].hist()


# In[12]:


data['log_diff'].hist()


# ### Test the data if stationary
# Eventhough the plot not show a obvious trend, we can run Augmented Dickey-Fuller test, which is a statistical method to testify if the time seris is stationary.
# * Null Hypothesis (H0): If failed to be rejected, it suggests the time series has a unit root, meaning it is non-stationary. It has some time dependent structure.
# * Alternate Hypothesis (H1): The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary. It does not have time-dependent structure.
# 
# We interpret this result using the p-value from the test. A p-value below a threshold (such as 5% or 1%) suggests we reject the null hypothesis (stationary), otherwise a p-value above the threshold suggests we fail to reject the null hypothesis (non-stationary).
# 
# * p-value > 0.05: Fail to reject the null hypothesis (H0), the data has a unit root and is non-stationary.
# * p-value <= 0.05: Reject the null hypothesis (H0), the data does not have a unit root and is stationary.

# In[13]:


def Augmented_DickeyFuller_test(x):
    result = adfuller(x)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

trans_data = data['log'].dropna()
Augmented_DickeyFuller_test(trans_data)


# Running the temperature data prints the test statistic value of -4. The more negative this statistic, the more likely we are to reject the null hypothesis (we have a stationary dataset).
# 
# As part of the output, we get a look-up table to help determine the ADF statistic. We can see that our statistic value of -4 is less than the value of -3.449 at 1%.
# 
# This suggests that we can reject the null hypothesis with a significance level of less than 1% (i.e. a low probability that the result is a statistical fluke).
# 
# Rejecting the null hypothesis means that the process has no unit root, and in turn that the time series is stationary or does not have time-dependent structure

# ## Task 2. Fit a simple moving average model (using the training set)

# In[14]:


def data_split(data):
    rows = len(data.index)
    train = data[:int(rows*0.7)].reset_index().drop(columns=['index']).iloc[:,0]
    test = data[int(rows*0.7)+1:].reset_index().drop(columns=['index']).iloc[:,0]
    return train, test

train, test = data_split(trans_data)


# ### Task 2.1 Apply the simple moving average model to the training data set, for a given k.

# In[15]:


def SMA(data, k):
    predict = data.rolling(window = k+1).mean()
    return predict


# Let the size of window is M. When calculating the SMA, the first M preidcts are not valid since M-th temperature data are required for the first moving average data point. As the result, when we calcualte the error, we only consider the data after M-th data.

# ### Task 2.2 Calculate the error, i.e., the difference between the predicted and original value in the training data set, and compute the root mean squared error (RMSE).
# ### Task 2.3 Repeat the above two steps by varying k and calculate the RMSE.

# In[16]:


def RMSE(actual, predict):
    return np.sqrt(mean_squared_error(actual, predict))
max_k = 100
windows = [i for i in range(1, max_k)]
sma_rmse = []

for k in windows:
    actual = train[k:]
    predict = SMA(train, k)[k:]
    sma_rmse.append(RMSE(actual, predict))


# ### Task 2.4 Plot RMSE vs k. Select k based on the lowest RMSE value. For the best value of k plot the predicted values against the original values.

# In[17]:


# Plot RMSE vs k.
plt.plot(windows, sma_rmse)
plt.grid()
plt.xlabel('Window Size(k)')
plt.ylabel('RMSE')
plt.title('SMA k selection')

print('Minumum RMSE value is {}, when k = {}'.format(min(sma_rmse), windows[sma_rmse.index(min(sma_rmse))]))


# In[18]:


def plot_org_vs_est(org, est, model, param):
    # Plot Original vs Predict
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(org, marker='.', label='Original')
    ax.plot(est, marker='*', label='Estimate')
    ax.legend()
    ax.set_ylabel('Temperature (F)')
    ax.set_xlabel('Time')
    ax.set_title('{} {} Original vs Estimate'.format(model, param))
    fig.savefig('{} {} Original vs Estimate.png'.format(model, param))


# In[19]:


# Knee point
k = 20
actual = train[k:]
predict = SMA(train, k)[k:]
plot_org_vs_est(actual, predict, 'SMA train data', 'k=20')


# In[20]:


k = 1
actual = train[k:]
predict = SMA(train, k)[k:]
# Plot Original vs Predict
plot_org_vs_est(actual, predict, 'SMA train data', 'k=1')


# ## Task 3. Fit an exponential smoothing model (use the training set)

# ### Task3.1 Apply the exponential smoothing mode to the training data set for α =0.1.

# In[21]:


def EMA(data, alpha=0.1):
    predict = [0] * len(data)
    predict[0] = data[0]
    for i in range(1,len(data)):
        predict[i] = alpha * data[i-1] + (1-alpha) * predict[i-1]
    return predict


# ### Task3.2 Calculate the error, i.e., the difference between the predicted and original value in the training data set, and compute the root mean squared error (RMSE).

# ### Task 3.3 Repeat steps 2.1 and 2.2 by increasing a each time by 0.1, until a = 0.9.

# In[22]:


ema_rmse = []
alpha = [i/10 for i in range(1,10)]
for a in alpha:
    predict = EMA(train,a)
    ema_rmse.append(RMSE(train, predict))


# ### Task 3.4 Plot RMSE vs a. Select a based on the lowest RMSE value.

# In[23]:


# Plot RMSE vs k.
plt.plot(alpha, ema_rmse)
plt.grid()
plt.xlabel('Alpha')
plt.ylabel('RMSE')
plt.title('EMA alpha selection')

print('Minumum RMSE value is {}, when alpha = {}'.format(min(ema_rmse), alpha[ema_rmse.index(min(ema_rmse))] ))


# ### Task 3.5 For the selected value of a plot the predicted values against the original values, and visually inspect the accuracy of the forecasting model.

# In[24]:


alpha = 0.1
actual = train.copy()
predict = EMA(train,alpha)
plot_org_vs_est(actual, predict, 'EMA train data', 'alpha=0.1')


# In[25]:


alpha = 0.9
actual = train.copy()
predict = EMA(train,alpha)
plot_org_vs_est(actual, predict, 'EMA train data', 'alpha=0.9')


# ## Task 4. Fit an AR(p) model (use the training set)

# ### Task 4.1 First select the order p of the AR model by plotting PACF in order to determine the lag k at which PACF cuts off, as discussed in section 6.4.4.

# In[26]:


plot_pacf(train,lags=30, title='PACF')

delta = 0.15
lag_pacf = pacf(train, nlags=30, method='yw')
upperInt = 1.96/np.sqrt(len(train))
intPoint = -1

for i in range(0,len(lag_pacf)):
    if abs(lag_pacf[i]-upperInt) <= delta:
        print("p value using PACF is " + str(i))
        p = i
        break


# ### Task 4.2 Estimate the parameters of the AR(p) model. Provide RMSE value and a plot the predicted values against the original values.

# In[27]:


AR = ARIMA(train, order=(p, 0, 0)) 
AR_fit = AR.fit()
print(AR_fit.summary())


# In[28]:


AR_RMSE = np.sqrt(np.mean(AR_fit.resid ** 2))
print('RSME of AR model with train data: {}'.format(AR_RMSE))
predict = AR_fit.fittedvalues
actual = train.copy()
plot_org_vs_est(actual, predict, 'AR train data', 'p={}'.format(p))


# ### Task 4.3 Carry out a residual analysis to verify the validity of the model.
# a. Do a Q-Q plot of the pdf of the residuals against N(0, s2) In addition, draw the residuals histogram and carry out a χ2 test that it follows the normal distribution N(0, s2).
# 
# b. Do a scatter plot of the residuals to see if there are any correlation trends.

# In[29]:


def plot_qqplot(error, title):
    sm.qqplot(error,loc=0,scale=np.sqrt(np.var(error)),line='q')
    plt.title('{} Q_Q plot'.format(title))
    plt.savefig('{} Q_Q plot.png'.format(title))
    plt.show()

def chisquare_test(x):
    print ("\nRunning chisquare test.........")
    k2, p = stats.normaltest(x)
    alpha = 0.05
    print('We have p-value is {}'.format(p))
    
    if p > alpha:
        print("Not Significant Result, failed to reject the null hypothesis")
    else:
        print("Significant Result, the null hypothesis rejected")

def plot_scatter(error, y_predict, title):
    plt.scatter(error,y_predict)
    plt.xlabel('residuals')
    plt.ylabel('y')
    plt.title('{} scatter plot.png'.format(title))
    plt.savefig('{} scatter plot.png'.format(title))
    plt.show()

# Residuals analysis
plot_qqplot(AR_fit.resid, 'AR residuals')
chisquare_test(AR_fit.resid)
plot_scatter(AR_fit.resid, AR_fit.fittedvalues, 'AR residuals')
AR_fit.resid.hist()


# ## Task 5. Comparison of all the models (use the testing set)

# In[30]:


k = 1
actual = test[k:]
predict = SMA(test, k)[k:]
# Plot Original vs Predict
plot_org_vs_est(actual, predict, 'SMA test data', 'k={}'.format(k))
SMA_RMSE = RMSE(actual, predict)
print('RSME of SMA model with test data: {}'.format(SMA_RMSE))


# In[31]:


alpha = 0.9
actual = test.copy()
predict = EMA(test,alpha)
plot_org_vs_est(actual, predict, 'EMA test data', 'alpha={}'.format(alpha))
EM_RMSE = RMSE(actual, predict)
print('RSME of EM model with test data: {}'.format(EM_RMSE))


# In[33]:


p = 3
AR = ARIMA(test, order=(p, 0, 0)) 
AR_fit = AR.fit()
AR_RMSE = np.sqrt(np.mean(AR_fit.resid ** 2))
print('RSME of AR model with test data: {}'.format(AR_RMSE))
predict = AR_fit.fittedvalues
actual = test.copy()
plot_org_vs_est(actual, predict, 'AR test data','p={}'.format(p))


# In[ ]:




