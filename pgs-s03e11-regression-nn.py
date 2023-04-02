#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sb
import lightgbm as lgb
import matplotlib.pyplot as plt
import math

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras import backend as K
from keras import models
from keras import layers
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

df = pd.read_csv('/kaggle/input/playground-series-s3e11/train.csv')

df.head(5)


# In[ ]:


# Sort by nulls in columns

pd.set_option('display.max_rows', df.shape[0])
pd.DataFrame(df.isnull().sum().sort_values(ascending = False))


# Lets see whats happend here

# I'll use two different regressors and one neural network to get result.

# In[ ]:


print(df.corr())

plt.figure(figsize = (20,20))
dataplot = sb.heatmap(df.corr(), cmap="YlGnBu", annot=True)
plt.show()


# Extracting labels, delete salad_bar, as copy of another feature, and delete id column
# 
# Scaling aas default

# In[ ]:


label = df['cost']

del df['id']
del df['cost']
del df['salad_bar']

scaler = StandardScaler()
train = pd.DataFrame(scaler.fit_transform(df),columns = df.columns)

train.head(5)


# Splitting data on test and train datasets, and splitting specialy for different models, with different random_states

# For NeuralNetwork model i normalize data by /100, after that i can use Dense Layer with single output, that returns 0-2.
# 
# To have actual prediction, i will bring pred back by 'x100.
# 
# yes, on test dataset it will make losses, but not so much
# Comfort of working is more important

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(
    train, label, test_size = 0.276, random_state = 1337)

y_train_d = y_train / 100
y_test_d = y_test / 100

x_train_lgbm, x_test_lgbm, y_train_lgbm, y_test_lgbm = train_test_split(
    train, label, test_size = 0.27, random_state = 37)

x_train_cat, x_test_cat, y_train_cat, y_test_cat = train_test_split(
    train, label, test_size = 0.25, random_state = 13)


# First model is LGBMRegressor. I've tested that regressor in another competition, and it works good

# In[ ]:


lgbm = LGBMRegressor(objective = 'regression', 
                     num_leaves = 1750,
                     max_depth = 40,
                     min_child_samples = 50,
                     learning_rate = 0.025, 
                     n_estimators = 114,
                     random_state = 1337).fit(x_train_lgbm, y_train_lgbm)
lgbm_train_predict = abs(lgbm.predict(x_test_lgbm))
rmsle = mean_squared_log_error(y_test_lgbm, lgbm_train_predict, squared=False)
print(rmsle)


# CatBoostRegressor works great, but it havent RMSLE as loss function
# And so, i have to make that loss function custom. realy, i've seen solution for that problem on the habr.com
# link on te post below, in code

# In[ ]:


class RMSLE(object):
#thnx to this post https://habr.com/ru/sandbox/163469/

    def calc_ders_range(self, approxes, targets, weights):
        assert len(approxes) == len(targets)
        if weights is not None:
            assert len(weights) == len(approxes)

        result = []
        for index in range(len(targets)):
            val = max(approxes[index], 0)
            der1 = math.log1p(targets[index]) - math.log1p(max(0, approxes[index]))
            der2 = -1 / (max(0, approxes[index]) + 1)

            if weights is not None:
                der1 *= weights[index]
                der2 *= weights[index]

            result.append((der1, der2))
        return result
    
class RMSLE_val(object):
    def get_final_error(self, error, weight):
        return np.sqrt(error / (weight + 1e-38))

    def is_max_optimal(self):
        return False

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        error_sum = 0.0
        weight_sum = 0.0

        for i in range(len(approx)):
            w = 1.0 if weight is None else weight[i]
            weight_sum += w
            error_sum += w * ((math.log1p(max(0, approx[i])) - math.log1p(max(0, target[i])))**2)

        return error_sum, weight_sum


cat = CatBoostRegressor(iterations = 100,
                          loss_function = RMSLE(),
                          learning_rate = 0.75,
                          verbose = 10,
                          eval_metric = RMSLE_val()).fit(x_train_lgbm, y_train_lgbm)
cat_train_predict = abs(cat.predict(x_test_lgbm))
rmsle_c = mean_squared_log_error(y_test_lgbm, cat_train_predict, squared=False)
print(rmsle_c)


# And NeuralNetwork, that fittin on downgraded lables by /100
# i've show RMSLE with bringed back lables

# In[ ]:


model = keras.Sequential([
       layers.Dense(512, activation="relu"),
       layers.Dense(256, activation="relu"),
       layers.Dropout(0.1),
       layers.Dense(128, activation="relu"),
       layers.Dropout(0.2),
       layers.Dense(64, activation="relu"),
       layers.Dense(32, activation="relu"),
       layers.Dense(16, activation="relu"),
       layers.Dense(1)
   ])

def root_mean_squared_log_error(y_true, y_pred):
       return K.sqrt(K.mean(K.square((K.log(y_pred + 1) - K.log(y_true + 1)))))

model.compile(optimizer="Adam",
            loss=root_mean_squared_log_error,
            metrics=["accuracy"])

model.fit(x_train, y_train_d, epochs = 20, batch_size = 128)

model_predict = abs(model.predict(x_test))
rmsle_1 = mean_squared_log_error(y_test_d, model_predict, squared=False)

model_predict *= 100
rmsle_2 = mean_squared_log_error(y_test, model_predict, squared=False)

print(rmsle_1)
print(rmsle_2)


# Preparing test data

# In[ ]:


test = pd.read_csv('/kaggle/input/playground-series-s3e11/test.csv')
test.head(5)


# In[ ]:


del test['id']
del test['salad_bar']

test = pd.DataFrame(scaler.fit_transform(test), columns = test.columns)


# I've made predictions on every model and find mean result of them
# This mean result i'll use in submission

# In[ ]:


lgbm_test_predict = abs(lgbm.predict(test)).reshape(-1, 1)
cat_test_predict = abs(cat.predict(test)).reshape(-1, 1)
model_test_predict = 100 * abs(model.predict(test)).reshape(-1, 1)

cost_0 = pd.DataFrame(data = lgbm_test_predict, columns = ['cost_0'])
cost_1 = pd.DataFrame(data = cat_test_predict, columns = ['cost_1'])
cost_2 = pd.DataFrame(data = model_test_predict, columns = ['cost_2'])
cost = pd.concat([cost_0, cost_1, cost_2], axis = 1)
cost = cost.mean(axis = 1)
cost = pd.DataFrame(data = cost, columns = ['cost'])
cost.head(5)


# In[ ]:


Id = pd.read_csv('/kaggle/input/playground-series-s3e11/test.csv')
submission = pd.DataFrame(Id['id'])
result = pd.concat([submission, cost], axis = 1)
result.to_csv('/kaggle/working/submission.csv', index = False)
result.head(10)

