# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.model_selection import RandomizedSearchCV
#from sklearn.preprocessing import normalize
from scipy.stats import uniform

file_name = 'databasek1k2flux.txt'
data = pd.read_csv(file_name, sep=" ", header=None)

data.columns = ['k1', 'k2', 'flux']


y = data.pop('flux')
X = data
#X = normalize(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# #normalize
X_mean = X_train.mean(axis = 0)
X_train -= X_mean
std = X_train.std(axis=0)
X_train /= std

X_test -= X_mean
X_test /= std

# choose layers
layers = (100, 10)

regr = MLPRegressor(hidden_layer_sizes= layers, batch_size = 1500, \
                    random_state=1,early_stopping=True,
                    max_iter=500).fit(X_train, y_train)
    
print('mlp regression score on test set %f\n'%(regr.score(X_test, y_test)))

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

# sample usage
save_object(regr, 'MLPregressor.pkl')
save_object(X_mean, 'X_mean.pkl')
save_object(std, 'std.pkl')


#attempt to tune hyperparameters 

# hyper_regr = MLPRegressor(early_stopping=True, hidden_layer_sizes=layers)

# distributions = dict(activation=['identity', 'logistic', 'tanh', 'relu'],\
#                      validation_fraction=[0.1,0.01],\
#                          solver=['sgd', 'adam'],\
#                         learning_rate=['constant', 'invscaling', 'adaptive'],\
#                         learning_rate_init=[0.001, 0.1, 0.01, 0.0001],\
#                             max_iter=[500, 800, 1000])

# clf = RandomizedSearchCV(hyper_regr, distributions, random_state=0)
# search = clf.fit(X_train, y_train)
# search.best_params_