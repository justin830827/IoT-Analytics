
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('./data/whu24.csv', names= ['x1', 'x2', 'x3', 'x4','x5', 'y'])
data.describe()


# In[3]:


X = data[['x1', 'x2', 'x3', 'x4','x5']]
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=2000, random_state=42)


# In[4]:


X_train.describe()


# In[5]:


X_test.describe()


# In[6]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[8]:


nn_oneLayer_tune = MLPRegressor(activation = 'logistic',
                                max_iter = 500,
                                solver = 'adam',
                                learning_rate = 'constant',
                                early_stopping = True,
                                random_state=42
                                )
parameters = {
    'hidden_layer_sizes': np.arange(18,26),
    'alpha': 10.0 ** -np.arange(1, 7),
    'learning_rate_init': 10.0 ** -np.arange(1, 7), 
}
clf_oneLayer = GridSearchCV(nn_oneLayer_tune, parameters, n_jobs = -1, verbose = 10)
clf_oneLayer.fit(X_train, y_train)


# In[9]:


score = []
score.append(clf_oneLayer.best_score_)
print("Best score: {}".format(clf_oneLayer.best_score_))
print("Best params: {}".format(clf_oneLayer.best_params_))


# In[11]:


loss = []
layers = []
nn_oneLayer = MLPRegressor(
                hidden_layer_sizes= clf_oneLayer.best_params_['hidden_layer_sizes'],
                activation = 'logistic',
                max_iter = 500,
                solver='adam',
                alpha= clf_oneLayer.best_params_['alpha'],
                learning_rate_init= clf_oneLayer.best_params_['learning_rate_init'],
                learning_rate= 'constant',
                early_stopping = True,
                random_state = 42
                ).fit(X_train, y_train)
layers.append(clf_oneLayer.best_params_['hidden_layer_sizes'])
loss.append(nn_oneLayer.loss_)
print("loss: {}".format(nn_oneLayer.loss_))


# In[12]:


# Tunning the second layer
nn_twoLayer_tune = MLPRegressor(activation = 'logistic',
                                max_iter = 500,
                                solver='adam',
                                learning_rate='constant',
                                early_stopping = True,
                                random_state = 42
                               )
parameters = {
    'hidden_layer_sizes': [ (21,i) for i in np.arange(12,24)],
    'alpha': 10.0 ** -np.arange(1, 7),
    'learning_rate_init': 10.0 ** -np.arange(1, 7), 
}
clf_twoLayer = GridSearchCV(nn_twoLayer_tune, parameters, n_jobs = -1, verbose = 10)
clf_twoLayer.fit(X_train, y_train)


# In[13]:


score.append(clf_twoLayer.best_score_)
print("Best score: {}".format(clf_twoLayer.best_score_))
print("Best params: {}".format(clf_twoLayer.best_params_))


# In[14]:


nn_twoLayer = MLPRegressor(hidden_layer_sizes= clf_twoLayer.best_params_['hidden_layer_sizes'],
                            activation = 'logistic',
                            max_iter = 500,
                            solver='adam',
                            alpha= clf_twoLayer.best_params_['alpha'],
                            learning_rate_init= clf_twoLayer.best_params_['learning_rate_init'],
                            learning_rate= 'constant',
                            early_stopping = True,
                            random_state = 42
                        ).fit(X_train, y_train) 

layers.append(clf_twoLayer.best_params_['hidden_layer_sizes'])
loss.append(nn_twoLayer.loss_)
print("loss: {}".format(nn_twoLayer.loss_))


# In[15]:


# Tunning the third layer
nn_threeLayer = MLPRegressor(
                            activation = 'logistic',
                            max_iter = 500,
                            solver='adam',
                            learning_rate='constant',
                            early_stopping = True,
                            random_state = 42
                            )
parameters = {
    'hidden_layer_sizes': [ (21,20,i) for i in np.arange(13,22)],
    'alpha': 10.0 ** -np.arange(1, 7),
    'learning_rate_init': 10.0 ** -np.arange(1, 7), 
}
clf_threeLayer = GridSearchCV(nn_threeLayer, parameters, n_jobs = -1, verbose = 10)
clf_threeLayer.fit(X_train, y_train)


# In[16]:


score.append(clf_threeLayer.best_score_)
print("Best score: {}".format(clf_threeLayer.best_score_))
print("Best params: {}".format(clf_threeLayer.best_params_))


# In[17]:


nn_threeLayer = MLPRegressor(hidden_layer_sizes= clf_threeLayer.best_params_['hidden_layer_sizes'],
                            activation = 'logistic',
                            max_iter = 500,
                            solver='adam',
                            alpha= clf_threeLayer.best_params_['alpha'],
                            learning_rate_init= clf_threeLayer.best_params_['learning_rate_init'],
                            learning_rate= 'constant',
                            early_stopping = True,
                            random_state = 42
                        ).fit(X_train, y_train) 
loss.append(nn_threeLayer.loss_)
print("loss: {}".format(nn_threeLayer.loss_))


# In[18]:


# Tunning the fourth layer
nn_fourLayer = MLPRegressor( activation = 'logistic',
                            max_iter = 500,
                            solver='adam',
                            learning_rate='constant',
                            early_stopping = True,
                            random_state = 42
                            )
parameters = {
    'hidden_layer_sizes': [(21,20,18,i) for i in range(12,20)],
    'alpha': 10.0 ** -np.arange(1, 7),
    'learning_rate_init': 10.0 ** -np.arange(1, 7), 
}
clf_fourLayer = GridSearchCV(nn_fourLayer, parameters, n_jobs = -1, verbose = 10)
clf_fourLayer.fit(X_train, y_train)


# In[19]:


score.append(clf_fourLayer.best_score_)
print("Best score: {}".format(clf_fourLayer.best_score_))
print("Best params: {}".format(clf_fourLayer.best_params_))


# In[20]:


# Tunning the fourth layer
nn_fourLayer = MLPRegressor(  hidden_layer_sizes= clf_fourLayer.best_params_['hidden_layer_sizes'],
                            activation = 'logistic',
                            max_iter = 500,
                            solver='adam',
                            alpha= clf_fourLayer.best_params_['alpha'],
                            learning_rate_init= clf_fourLayer.best_params_['learning_rate_init'],
                            learning_rate= 'constant',
                            early_stopping = True,
                            random_state = 42
                        ).fit(X_train, y_train)
loss.append(nn_fourLayer.loss_)
print("loss: {}".format(nn_fourLayer.loss_))


# In[21]:


from sklearn.metrics import mean_squared_error
test_loss = []
y_predict = nn_oneLayer.predict(X_test)
test_loss.append(mean_squared_error(y_test, y_predict))
y_predict = nn_twoLayer.predict(X_test)
test_loss.append(mean_squared_error(y_test, y_predict))
y_predict = nn_threeLayer.predict(X_test)
test_loss.append(mean_squared_error(y_test, y_predict))
y_predict = nn_fourLayer.predict(X_test)
test_loss.append(mean_squared_error(y_test, y_predict))
print("Train loss: {}".format(loss))
print("Test loss: {}".format(test_loss))


# In[22]:


plt.plot(np.arange(1,len(loss)+1),loss, marker='o', color='orange', linewidth=2, label= 'train loss')
plt.plot(np.arange(1,len(loss)+1),test_loss, marker='o', color='blue', linewidth=2, label= 'test loss')
plt.xlabel('Number of layers')
plt.ylabel('Mean Squared Error(MSE)')
plt.title('Training data vs Testing data')
# plt.axvline(x=4, linestyle='--', color='red')
plt.savefig('Training data vs Testing data.png')            
plt.legend()
plt.show()


# In[23]:


# Regression model
import statsmodels.api as sm
X_train = sm.add_constant(X_train)
model = sm.OLS(y,X).fit()
print (model.summary())
y_predict_regess = model.predict(X_test)


# In[36]:


# Best ANN model
y_predict_ann = nn_threeLayer.predict(X_test)


# In[37]:


def SSE(y_true, y_predict):
    y_true = y_true.to_numpy()
    if len(y_true) != len(y_predict):
        return
    sse = 0
    for i in range(len(y_true)):
        sse += (y_true[i] - y_predict[i]) ** 2
    return sse

ann_sse = SSE(y_test,y_predict_ann)
regress_sse = SSE(y_test,y_predict_regess)


# In[38]:


print ("Best ANN's SSE: {}".format(ann_sse))
print ("Best Regression's SSE: {}".format(regress_sse))

