from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import tensorflow as tf
from datetime import datetime
from datetime import date
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('cars.csv')
dataset = dataset.drop(columns=['Images'])
dataset.head()

# converting registration date to age of car in days
today = date.today()

dates = dataset.iloc[:, 4].values

age_in_days = []

for i in dates:
    d1 = datetime.strptime(i, "%d-%b-%Y").date()
    delta = today - d1
    difference = delta.days
    age_in_days.append(difference)

dataset['Age'] = age_in_days

dataset = dataset.drop(columns=['Registration Date'])

# converting category to integer values

models = dataset['Car Model'].values
brands = dataset['Brand'].values
categories = dataset['Category'].values

encoder = LabelEncoder()
models_data = encoder.fit_transform(models)
brands_data = encoder.fit_transform(brands)
categories_data = encoder.fit_transform(categories)

dataset = dataset.drop(columns=['Car Model'])
dataset = dataset.drop(columns=['Brand'])
dataset = dataset.drop(columns=['Category'])

dataset['Car Model'] = models_data
dataset['Brand'] = brands_data
dataset['Category'] = categories_data

# splitting the data

X = dataset.iloc[:, 1:7]
y = dataset.iloc[:, 0]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# training the model


linReg = LinearRegression()

linReg.fit(X_train, y_train)

# Saving model to disk
pickle.dump(linReg, open('model.pkl', 'wb'))
