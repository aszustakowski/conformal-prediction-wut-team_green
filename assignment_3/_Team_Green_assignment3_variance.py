#!/usr/bin/env python
# coding: utf-8

# In[1]:


#wczytanie niezbędnych bibliotek
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import catboost
from numpy import random
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#wczytanie zbudowanego modelu
model = catboost.CatBoostClassifier()
model.load_model('./model.cbm')


# In[5]:


#wczytanie początkowych (niezaburzonych) danych
data_train = pd.read_csv('./train_kaggle_raw.csv')
data_test = pd.read_csv('./test_kaggle_raw.csv')
X_train = data_train.drop('TARGET', axis=1)
y_train = data_train['TARGET']
X_test = data_test.drop('TARGET', axis=1)
y_test = data_test['TARGET']


# In[ ]:


get_ipython().run_cell_magic('time', '', '#pierwsze dopasowanie modelu dla początkowych danych\nmodel.fit(X_train, y_train)\nmodel_pred_prob = pd.DataFrame({"start": model.predict_proba(X_test)[:,0]})\n\n#wybranie ciągłych zmiennych do zaburzania\nindexes = [0,1,2,3,4,5,11,12,13,17,19,20,21,23]\n\n#pętla narzucająca szum na dane wejściowe, trenująca na nich model i tworząca macierz z prawdopodobieństwem należności do klasy 1 dla każdej obserwacji\nfor j in range(10):\n    i=0\n    for name in list(X_train.columns):\n        modified_column = X_train[name]+ np.random.normal(loc = X_train.mean()[i], scale = abs(X_train.mean()[i])*10, size = X_train[name].shape)\n        if i==0:\n            df = pd.DataFrame({name: modified_column})\n        elif i in indexes:\n            df.insert(i, name, modified_column)\n        else:\n            df.insert(i, name, X_train[name])\n        i=i+1\n    model.fit(df, y_train)\n    model_pred_prob.insert(j, j, model.predict_proba(X_test)[:,0])\n')


# In[11]:


#transpozycja macierzy i wyciągnięcie wariancji oraz średniej każdej kolumny (każdej obserwacji)
model_pred_prob_transposed = model_pred_prob.T
variations = model_pred_prob_transposed.var()
scores = model_pred_prob_transposed.mean()


# In[7]:


#stworzenie przedziałów ufności dla każdej z obserwacji
n = len(X_train)
alpha = 0.01
qhat = np.quantile(conformity_scores, np.ceil((n+1)*(1-alpha))/n)
#x = np.unique(X_test)
#x_odds = x/(1-x)
output = np.array([scores-qhat*variations,scores+qhat*variations])


# In[13]:


#wizualizacja przedziałów
indices_2 = X_test.index
k = 25
variable = 'REGION_POPULATION_RELATIVE'
for i in range(k):
    obs_num = random.randint(0,len(X_test)-1)
    plt.scatter(X_test[variable][indices_2[obs_num]],scores[indices_2[obs_num]], color = 'blue')
    plt.scatter(X_test[variable][indices_2[obs_num]],output[0, obs_num], color = 'r')
    plt.scatter(X_test[variable][indices_2[obs_num]],output[1, obs_num],  color = 'lime')
    plt.plot((X_test[variable][indices_2[obs_num]], X_test[variable][indices_2[obs_num]]), (output[0, obs_num], scores[indices_2[obs_num]]), color = 'r', linestyle = '--', alpha = 0.5)
    plt.plot((X_test[variable][indices_2[obs_num]], X_test[variable][indices_2[obs_num]]), (scores[indices_2[obs_num]], output[1, obs_num]), color = 'lime', linestyle = '--', alpha = 0.5)
obs_num = random.randint(0,len(X_test)-1)

plt.scatter(X_test[variable][indices_2[obs_num]],scores[indices_2[obs_num]], color = 'blue',label = 'Prediction')
plt.scatter(X_test[variable][indices_2[obs_num]],output[0, obs_num], label = 'Upper bound prediction', color = 'r')
plt.scatter(X_test[variable][indices_2[obs_num]],output[1, obs_num], label = 'Lower bound prediction', color = 'lime')
plt.plot((X_test[variable][indices_2[obs_num]], X_test[variable][indices_2[obs_num]]), (output[0, obs_num], scores[indices_2[obs_num]]), color = 'r', linestyle = '--', alpha = 0.5)
plt.plot((X_test[variable][indices_2[obs_num]], X_test[variable][indices_2[obs_num]]), (scores[indices_2[obs_num]], output[1, obs_num]), color = 'lime', linestyle = '--', alpha = 0.5)

plt.legend()
plt.xlabel(f'Logit {variable} value')
plt.ylabel('Odd scores')
#plt.yscale('logit')
plt.xscale('logit')
plt.title(f'Conformal regression interval for selected {k} test observations')
plt.show()


# In[ ]:




