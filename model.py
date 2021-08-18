import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
import seaborn as sns
import pickle
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('carsv3.csv')
dataset = dataset.drop(columns=['Images','Car Model'])
dataset

# converting registration date to age of car in days
today = date.today()

dates = dataset.iloc[:, 3].values

age_in_days = []

for i in dates:
    d1 = datetime.strptime(i, "%d-%b-%Y").date()
    delta = today - d1
    difference = delta.days
    age_in_days.append(difference)

dataset['Age'] = age_in_days

dataset = dataset.drop(columns=['Registration Date'])
dataset.head()
#show data info
dataset.shape
dataset.describe().transpose()
dataset.info()


#examine outliers numerical features)
cols=['Depreciation','Engine Capacity', 'Mileage']
sns.boxplot(dataset[cols[0]])

sns.boxplot(dataset[cols[1]])

sns.boxplot(dataset[cols[2]])

def find_outliers_limit(dataset,col):
    print(col)
    print('-'*50)
    #removing outliers
    q25, q75 = np.percentile(dataset[col], 25), np.percentile(dataset[col], 75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower = q25 - cut_off
    upper = q75 + cut_off
    print('Lower:',lower,' Upper:',upper)
    return lower,upper
def remove_outlier(dataset,col,upper,lower):
    # identify outliers
    outliers = [x for x in dataset[col] if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    # remove outliers
    outliers_removed = [x for x in dataset[col] if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(outliers_removed))
    final= np.where(dataset[col]>upper,upper,np.where(dataset[col]<lower,lower,dataset[col]))
    return final
outlier_cols=['Depreciation','Engine Capacity','Mileage']
for col in outlier_cols:
    lower,upper=find_outliers_limit(dataset,col)
    dataset[col]=remove_outlier(dataset,col,upper,lower)

#visual any outlier after reduction
plt.figure(figsize=(20,10))
dataset[outlier_cols].boxplot()


# Mileage and engine capacity is continous variable, spilt into bin 
labels=[0,1,2,3,4,5,6,7,8,9]
dataset['Mileage_bin']=pd.cut(dataset['Mileage'],len(labels),labels=labels)
dataset['Mileage_bin']=dataset['Mileage_bin'].astype(float)
labels=[0,1,2,3,4]
dataset['EC_bin']=pd.cut(dataset['Engine Capacity'],len(labels),labels=labels)
dataset['EC_bin']=dataset['EC_bin'].astype(float)


#use ordinal encoder to handle categorical columns
num_dataset=dataset.select_dtypes(include=np.number)

cat_dataset=dataset.select_dtypes(include=object)

encoding=OrdinalEncoder()

cat_cols=cat_dataset.columns.tolist()

encoding.fit(cat_dataset[cat_cols])

cat_oe=encoding.transform(cat_dataset[cat_cols])

cat_oe=pd.DataFrame(cat_oe,columns=cat_cols)

cat_dataset.reset_index(inplace=True,drop=True)

cat_oe.head()

num_dataset.reset_index(inplace=True,drop=True)

cat_oe.reset_index(inplace=True,drop=True)

final_all_dataset=pd.concat([num_dataset,cat_oe],axis=1)

final_all_dataset

#scale data and spilt data 
cols_drop=['Ask Price','price_log']
X=final_all_dataset.drop(cols_drop,axis=1)
y=final_all_dataset['Ask Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
print(X_train_scaled.shape) # x_train and x_test shape should be 2D array
print(y_train.shape)

#train the model using 1000 decision trees
model=RandomForestRegressor(n_estimators = 1000, random_state = 42)
model.fit(X_train_scaled,y_train)

# Use the forest's predict method on the test data
y_pred = model.predict(X_test_scaled)


# Saving model to disk
pickle.dump(model, open('model.pkl', 'rb'))
