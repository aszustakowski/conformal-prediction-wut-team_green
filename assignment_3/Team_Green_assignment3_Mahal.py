#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import random
#zaczytanie danych
train = pd.read_csv(r"C:\Users\AntoSzu\Downloads\data_green\train_green.csv")
test = pd.read_csv(r"C:\Users\AntoSzu\Downloads\data_green\test_green.csv")


# In[2]:


#w zadaniu 3 X to zmienne predykcyjne z modelu
X_train = train.drop(['PRED_BINNED', 'SEG_1', 'SEG_2', 'SEG_3'], axis = 1)
pred_train = train['PRED']
y_train = train['TARGET']
X_test = test.drop(['PRED_BINNED', 'SEG_1', 'SEG_2', 'SEG_3'], axis = 1)
pred_test = test['PRED']
y_test= test['TARGET']
#iloraz szans
train_odds = (pred_train)/(1-pred_train)


# In[3]:


#ignorowanie warningów dla notebooka (df_approx['score_diff'] = np.abs(df_approx['PRED']-X_train[i]) produkuje warning)
import warnings
warnings.filterwarnings('ignore')


# ## MAHALANOBIS DISTANCE

# In[4]:


#źródło: https://www.geeksforgeeks.org/how-to-calculate-mahalanobis-distance-in-python/
def calculateMahalanobis(y=None, data=None, cov=None): 
  
    y_mu = y - np.mean(data) 
    if not cov: 
        cov = np.cov(data.values.T) 
    inv_covmat = np.linalg.inv(cov) 
    left = np.dot(y_mu, inv_covmat) 
    mahal = np.dot(left, y_mu.T) 
    return mahal
# .diagonal() 


# In[5]:


#dla uproszeczenia do odległości mahalanobisa pozbędziemy się obserwacji z brakami danych
df_mahal_train = X_train.dropna()
pred_mahal_train = df_mahal_train['PRED']
train_odds_mahal = (pred_mahal_train)/(1-pred_mahal_train)
df_mahal_test = X_test.dropna()
pred_mahal_test = df_mahal_test['PRED']


# In[6]:


get_ipython().run_cell_magic('time', '', "#przybliżamy prawdopodobieństwo wystąpienia klasy 1 (to będzie nasz y)\n#liczba obserwacji służąca do przybliżenia\nm = 1000\n#liczba obserwacji bez NA (dla uproszczenia żeby odległość mahalanobisa była wyliczalna)\nn = len(df_mahal_train)\n#conformity scores dla ilorazu szans\nconformity_scores = []\n#odległość mahalanobisa\nmahalanobis_distances = []\n#indeksy obserwacji, które nie zawierają NA na żadnej zmiennej\nindices = df_mahal_train.index\nfor i in range(n):\n    df_approx = df_mahal_train[['PRED', 'TARGET']]\n    df_approx['score_diff'] = np.abs(df_approx['PRED']-df_approx['PRED'][indices[i]])\n    a = df_approx.nsmallest(m, 'score_diff')\n    indices_temp = np.array(a.index)\n    mean_target = np.mean(a['TARGET'])\n    mean_odds = mean_target/(1-mean_target)\n    conformity_scores.append(np.abs(train_odds_mahal[indices[i]]-mean_odds))\n    \n    #tworzymy pomocniczą ramkę danych zawierającą 1000 obserwacji o scorze najbliższym rozpatrywanej obserwacji\n    a = df_mahal_train.drop(['PRED', 'TARGET'], axis = 1)\n    a = a.loc[indices_temp,:]\n    #na zerowym miejscu zawsze będzie rozpatrywana aktualnie obserwacja\n    #wyznaczamy odległość mahalanobisa od średniej zmiennych predykcyjnych rozkładu wokół danego score'a\n    mahalanobis_distances.append(calculateMahalanobis(y = a.iloc[0], data = a))\n")


# In[7]:


get_ipython().run_cell_magic('time', '', "#analogiczna pętla służąca do wyznaczenia odległości mahalanobisa obserwacji testowej od średniej zmiennych predykcyjnych\n#dla 1000 najbliższych scorów\n\n#liczba obserwacji służąca do przybliżenia\nm = 1000\n#liczba obserwacji testowych, które nie zawierają NA\nn_test = len(df_mahal_test)\n#indeksy obserwacji, które nie zawierają NA na żadnej zmiennej\nindices_2 = df_mahal_test.index\nmahalanobis_test = []\n#ramka danych ze zmiennymi predykcyjnymi dla obserwacji ze zbioru testowego\nb = df_mahal_test.drop(['PRED', 'TARGET'], axis = 1)\nfor i in range(n_test):\n    #grupowanie po scorach najbliższych dla danej obserwacji\n    df_approx = df_mahal_train[['PRED', 'TARGET']]\n    df_approx['score_diff'] = np.abs(df_approx['PRED']-df_mahal_test['PRED'][indices_2[i]])\n    a = df_approx.nsmallest(m, 'score_diff')\n    indices_temp = np.array(a.index)\n    #odległość mahalanobisa obserwacji od 1000 najbliższych obserwacji ze zbioru kalibracyjnego\n    a = df_mahal_train.drop(['PRED', 'TARGET'], axis = 1)\n    a = a.loc[indices_temp,:]\n    mahalanobis_test.append(calculateMahalanobis(y = b.iloc[i], data = a))\n")


# In[8]:


get_ipython().run_cell_magic('time', '', '#wyznaczamy unormowaną rangę odległości mahalanobisa dla obserwacji ze zbioru testowego względem "typowych" odległości\n#dla zbioru kalibracyjnego\nmahalanobis_rank = []\nfor i in range(n_test):\n    mahalanobis_temp = mahalanobis_distances.copy()\n    mahalanobis_temp.append(mahalanobis_test[i])\n    mahalanobis_rank.append(scipy.stats.rankdata(mahalanobis_temp)[n]/n)\n')


# In[10]:


#histogram zmienności dla zbioru testowego
plt.hist(mahalanobis_rank)
plt.title('Histogram of variance of test set samples, based on Mahalanobis Distance')
plt.show()


# In[13]:


#conformity scores dla zbioru kalibracyjnego
#conformity_scores = np.maximum(X_train-y_train, y_train-X_train)
#zadany poziom alpha
alpha = 0.01
#liczność próby
#n = len(X_train)
#kwantyl scorów
qhat = np.quantile(conformity_scores, np.ceil((n+1)*(1-alpha))/n)
#regresja dla zbioru testowego
test_odds = (pred_test[indices_2])/(1-pred_test[indices_2])
#przedział dla każdego punktu testowego
output_mahalanobis_lower = []
output_mahalanobis_upper = []
for i in range(n_test):
    output_mahalanobis_lower.append(test_odds[indices_2[i]]-qhat*mahalanobis_rank[i])
    output_mahalanobis_upper.append(test_odds[indices_2[i]]+qhat*mahalanobis_rank[i])


# In[24]:


#ile wartości jest unikatowych dla danej zmiennych (aby łatwiej wybrać zmienne do )
for i in df_mahal_test.columns:
    print(f'Zmienna {i}, frakcja unikatowych wartości {len(df_mahal_test[i].unique())/n_test}')


# In[26]:


k = 25
variable = 'FEAT_2'
for i in range(k):
    #losujemy indeks obserwacji
    obs_num = random.randint(0,n_test-1)
    #rysujemy predykcję
    plt.scatter(df_mahal_test[variable][indices_2[obs_num]],test_odds[indices_2[obs_num]], color = 'blue')
    #rysujemy górny/dolny kraniec przedziału ufności
    plt.scatter(df_mahal_test[variable][indices_2[obs_num]],output_mahalanobis_lower[obs_num], color = 'r')
    plt.scatter(df_mahal_test[variable][indices_2[obs_num]],output_mahalanobis_upper[obs_num],  color = 'lime')
    #łączymy kreską
    plt.plot((df_mahal_test[variable][indices_2[obs_num]], df_mahal_test[variable][indices_2[obs_num]]), (output_mahalanobis_lower[obs_num], test_odds[indices_2[obs_num]]), color = 'r', linestyle = '--', alpha = 0.5)
    plt.plot((df_mahal_test[variable][indices_2[obs_num]], df_mahal_test[variable][indices_2[obs_num]]), (test_odds[indices_2[obs_num]], output_mahalanobis_upper[obs_num]), color = 'lime', linestyle = '--', alpha = 0.5)

#powtarzamy krok z pętli po raz ostatni, aby dodać label do punktów
obs_num = random.randint(0,n_test-1)
plt.scatter(df_mahal_test[variable][indices_2[obs_num]],test_odds[indices_2[obs_num]], color = 'blue',label = 'Prediction')
plt.scatter(df_mahal_test[variable][indices_2[obs_num]],output_mahalanobis_lower[obs_num], label = 'Upper bound prediction', color = 'r')
plt.scatter(df_mahal_test[variable][indices_2[obs_num]],output_mahalanobis_upper[obs_num], label = 'Lower bound prediction', color = 'lime')
plt.plot((df_mahal_test[variable][indices_2[obs_num]], df_mahal_test[variable][indices_2[obs_num]]), (output_mahalanobis_lower[obs_num], test_odds[indices_2[obs_num]]), color = 'r', linestyle = '--', alpha = 0.5)
plt.plot((df_mahal_test[variable][indices_2[obs_num]], df_mahal_test[variable][indices_2[obs_num]]), (test_odds[indices_2[obs_num]], output_mahalanobis_upper[obs_num]), color = 'lime', linestyle = '--', alpha = 0.5)

plt.legend()
plt.xlabel(f'Logit {variable} value')
plt.ylabel('Odd scores')
plt.xscale('logit')
plt.title(f'Conformal regression interval for selected {k} test observations')
plt.show()

