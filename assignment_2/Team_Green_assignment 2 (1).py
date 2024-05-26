#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#zaczytanie danych
train = pd.read_csv(r"C:\Users\AntoSzu\Downloads\data_green\train_green.csv")
test = pd.read_csv(r"C:\Users\AntoSzu\Downloads\data_green\test_green.csv")
#w zadaniu 2 X to predykcja z modelu
X_train = train['PRED']
y_train = train['TARGET']
X_test = test['PRED']
y_test= test['TARGET']
#iloraz szans
X_train_odds = (X_train)/(1-X_train)


# In[26]:


#ignorowanie warningów dla notebooka (df_approx['score_diff'] = np.abs(df_approx['PRED']-X_train[i]) produkuje warning)
import warnings
warnings.filterwarnings('ignore')


# ## MARGINAL

# In[29]:


get_ipython().run_cell_magic('time', '', "#przybliżamy prawdopodobieństwo wystąpienia klasy 1 (to będzie nasz y)\n#liczba obserwacji służąca do przybliżenia\nm = 1000\nn = len(X_train)\n#conformity scores dla ilorazu szans\nconformity_scores = []\nfor i in range(n):\n    df_approx = train[['PRED', 'TARGET']]\n    df_approx['score_diff'] = np.abs(df_approx['PRED']-X_train[i])\n    a = df_approx.nsmallest(m, 'score_diff')\n    mean_target = np.mean(a['TARGET'])\n    mean_odds = mean_target/(1-mean_target)\n    conformity_scores.append(np.abs(X_train_odds[i]-mean_odds))\n")


# In[64]:


#conformity scores dla zbioru kalibracyjnego
#conformity_scores = np.maximum(X_train-y_train, y_train-X_train)
#zadany poziom alpha
alpha = 0.01
#liczność próby
#n = len(X_train)
#kwantyl scorów
qhat = np.quantile(conformity_scores, np.ceil((n+1)*(1-alpha))/n)
#regresja dla zbioru testowego
#przedział jedynie dla unikatowych wartości predykcji
x = np.unique(X_test)
x_odds = x/(1-x)
output = np.array([x_odds-qhat,x_odds+qhat])


# In[65]:


#wykres przedziałów ze skalą liniową
plt.plot(x_odds,x_odds, label = 'Prediction')
plt.fill_between(x_odds, output[0,:], output[1,:], color='r', alpha=.25, label = f'Prediction interval {np.round(1-alpha,2)} level')
plt.plot(x_odds,output[0,:], label = 'Upper bound prediction', color = 'r', linestyle = '--', alpha = 0.5)
plt.plot(x_odds,output[1,:], label = 'Lower bound prediction', color = 'r', linestyle = '--', alpha = 0.5)
plt.legend(loc = 'upper right')
plt.xlabel('Odd scores')
plt.ylabel('Odd scores')
plt.title('Conformal regression intervals for prediction odd scores on test set')
plt.show()


# In[66]:


#wykres przedziałów ze skalą logitową
def logit_hist(x):
    return np.log10(x/(1-x))
plt.plot(logit_hist(x_odds),x_odds, label = 'Prediction')
plt.fill_between(logit_hist(x_odds), output[0,:], output[1,:], color='r', alpha=.25, label = f'Prediction interval {np.round(1-alpha,2)} level')
plt.plot(logit_hist(x_odds),output[0,:], label = 'Upper bound prediction', color = 'r', linestyle = '--', alpha = 0.5)
plt.plot(logit_hist(x_odds),output[1,:], label = 'Lower bound prediction', color = 'r', linestyle = '--', alpha = 0.5)
plt.legend(loc = 'upper right')
plt.xlabel('Logit odd scores')
plt.ylabel('Odd scores')
plt.title('Conformal regression intervals for prediction odd scores on test set')
plt.show()

