# -*- coding: utf-8 -*-
"""Capstone (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1g-Wz7qkfpYE7riENfLF64-zTFc3C0QqU
"""

# Commented out IPython magic to ensure Python compatibility.
#importing the packages  
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# %matplotlib inline

#import the dataset
train = pd.read_excel('train.xlsx')
test = pd.read_excel('test.xlsx')

train.head()

train1 = train

train1.head()

train.drop('customerID', axis=1, inplace=True)

train.isnull().sum()

train.info()

train['TotalCharges'] = pd.to_numeric(train['TotalCharges'], errors='coerce')

train.columns

cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod','Churn']

num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

train.describe()

sns.boxplot(data=train1[['tenure', 'MonthlyCharges', 'TotalCharges']])
plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in cat_cols:
    train[col]= le.fit_transform(train[col])

# Create new features
train['MonthlyChargesPerTenure'] = train['MonthlyCharges'] / train['tenure']
train['TotalChargesPerTenure'] = train['TotalCharges'] / train['tenure']

sns.scatterplot(x=train['MonthlyChargesPerTenure'], y=train['TotalChargesPerTenure'])
plt.show()

train.shape

train=train.dropna()
train=train.drop_duplicates()

train.shape

train.head()

"""## **Data Visualization**"""

# create a histogram of a numeric feature
sns.histplot(train1['tenure'])
plt.show()

# create a scatter plot of two numeric features
sns.scatterplot(data=train1, x='MonthlyCharges', y='TotalCharges')
plt.show()

ax = sns.distplot(train1['tenure'], hist=True, kde=False, 
             bins=int(180/5), color = 'red', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
ax.set_ylabel('# of Customers')
ax.set_xlabel('Tenure (months)')
ax.set_title('# of Customers by their tenure')

sns.histplot(train1['MonthlyCharges'], kde=False, bins=30)
plt.show()

sns.histplot(train1['TotalCharges'], kde=False, bins=30)
plt.show()

# Set plot style
sns.set_style("whitegrid")

# Create a bar plot of gender distribution
fig, ax = plt.subplots()
sns.countplot(x='gender', data=train1, palette='Set2')
ax.set_xlabel('Gender')
ax.set_ylabel('Count')
ax.set_title('Gender Distribution')

sns.histplot(train1['MonthlyCharges'], bins=30)
plt.title('Distribution of Monthly Charges')
plt.show()

# Create a bar plot of gender distribution
fig, ax = plt.subplots()
sns.countplot(x='SeniorCitizen', data=train1, palette='Set2')
ax.set_xlabel('SeniorCitizen')
ax.set_ylabel('Count')
ax.set_title('SeniorCitizen Distribution')


# Add percentage labels to the plot
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height+3, '{:.1f}%'.format(100 * height/len(train1)),
            ha="center", fontsize=12, weight='bold')

ax = train1['Contract'].value_counts().plot(kind = 'bar',rot = 0, width = 0.3)
ax.set_ylabel('# of Customers')
ax.set_title('# of Customers by Contract Type')

sns.violinplot(data=train1, x='Contract', y='tenure', palette='Blues')
plt.xlabel('Contract Type')
plt.ylabel('Tenure (months)')
plt.title('Distribution of Tenure by Contract Type')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
fig, ax = plt.subplots(figsize=(8, 6))

for Contract in train1['Contract'].unique():
    sns.distplot(train1[train1['Contract']==Contract]['tenure'], hist=False,
                 label=Contract, ax=ax, kde_kws={'linewidth': 2})

ax.set_xlabel('Tenure (months)')
ax.set_ylabel('Density')
ax.set_title('Distribution of Tenure by Contract Type')
ax.legend()
plt.show()

# calculate the percentage of senior citizens
senior_pct = 100 * train1['SeniorCitizen'].value_counts(normalize=True)[1]

# create a horizontal bar chart
fig, ax = plt.subplots(figsize=(5, 3))
ax.barh(['Non-Senior', 'Senior'], [100-senior_pct, senior_pct])
ax.set_xlabel('Percentage')
ax.set_title('% of Senior Citizens')
plt.show()

import seaborn as sns

# list of categorical features
services = ['PhoneService','MultipleLines','InternetService','OnlineSecurity', 'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']

# create a grid of bar plots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 12))
for i, service in enumerate(services):
    row = i // 3
    col = i % 3
    ax = sns.countplot(x=service, data=train1, ax=axes[row, col])
    ax.set_title(service)

plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.ticker as mtick

# Create a pivot table with Contract and Churn columns
contract_churn =  train1.pivot_table(index='Contract', columns='Churn', aggfunc='size')

# Calculate the percentage of customers in each category
contract_churn_perc = (contract_churn.T / contract_churn.sum(axis=1)).T * 100
colors = ['#b0ceff', '#a10089']
# Plot the stacked bar chart with data labels
ax = contract_churn_perc.plot(kind='bar', stacked=True, color=colors, width=0.3, rot=0, figsize=(10,6))
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='best', prop={'size': 14}, title='Churn')
ax.set_ylabel('% Customers', size=14)
ax.set_title('Churn by Contract Type', size=14)

# Add data labels to each section of the stacked bar chart
for i, category in enumerate(contract_churn.columns):
    for j, value in enumerate(contract_churn_perc[category]):
        ax.text(j, contract_churn_perc.iloc[j, :i].sum() + value/2, '{:.0f}%'.format(value),
                ha='center', va='center', color='white', weight='bold', size=14)

"""From above graph we can observe that most of the customers are youngsters """

test.head()

test.columns

test.isnull().sum()

test.info()

cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod']

num_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for col in cat_cols:
    test[col]= le.fit_transform(test[col])

# Create new features
test['MonthlyChargesPerTenure'] = test['MonthlyCharges'] / test['tenure']
test['TotalChargesPerTenure'] = test['TotalCharges'] / test['tenure']

test.shape

test=test.dropna()
test=test.drop_duplicates()

test.shape

X = train.drop('Churn', axis=1)
y = train['Churn']

selector = SelectKBest(chi2, k=10)
X_new = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]
train_X = train[selected_features]
test_new = selector.transform(test)
# convert test_new from numpy array to list
test_new_list = test_new.tolist()
# create new DataFrame from test_new_list
test = pd.DataFrame(test_new_list, columns=selected_features)

train_X.columns

data = pd.concat([train_X, y], axis=1)

data.head()

from google.colab import files

data.to_csv('train.csv', index=False)

# replace "df.csv" with your file name
files.download("train.csv")

train_X['MonthlyCharges'].unique()

test.head()

test['Contract'].unique()

from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
X=pd.DataFrame(min_max.fit_transform(train_X))
test=pd.DataFrame(min_max.fit_transform(test))
y_array = y.values.reshape(-1, 1)
y = pd.DataFrame(min_max.fit_transform(y_array))

X.head()

X.shape

test.head()

y.head()

# Create a bar plot of gender distribution
fig, ax = plt.subplots()
sns.countplot(x=0, data=y, palette='Set2')
ax.set_xlabel('Churn')
ax.set_ylabel('Count')
ax.set_title('Churn Distribution')


# Add percentage labels to the plot
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height+3, '{:.1f}%'.format(100 * height/len(y)),
            ha="center", fontsize=12, weight='bold')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from imblearn.over_sampling import RandomOverSampler

# Instantiate the RandomOverSampler
ros = RandomOverSampler(random_state=42)

# Fit the sampler to the training data and balance the classes
X_train, y_train = ros.fit_resample(X_train, y_train)
X_test, y_test = ros.fit_resample(X_train, y_train)

# Create a bar plot of gender distribution
fig, ax = plt.subplots()
sns.countplot(x=0, data=y_train, palette='Set2')
ax.set_xlabel('Churn')
ax.set_ylabel('Count')
ax.set_title('Churn Distribution')


# Add percentage labels to the plot
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2., height+3, '{:.1f}%'.format(100 * height/len(y_train)),
            ha="center", fontsize=12, weight='bold')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Instantiate the models
lr = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
svm = SVC()

# Train each model on the same training data
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Evaluate the performance of each model on the same test data
y_pred_lr = lr.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_svm = svm.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_dt = accuracy_score(y_test, y_pred_dt)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_svm = accuracy_score(y_test, y_pred_svm)

prec_lr = precision_score(y_test, y_pred_lr)
prec_dt = precision_score(y_test, y_pred_dt)
prec_rf = precision_score(y_test, y_pred_rf)
prec_svm = precision_score(y_test, y_pred_svm)

rec_lr = recall_score(y_test, y_pred_lr)
rec_dt = recall_score(y_test, y_pred_dt)
rec_rf = recall_score(y_test, y_pred_rf)
rec_svm = recall_score(y_test, y_pred_svm)

f1_lr = f1_score(y_test, y_pred_lr)
f1_dt = f1_score(y_test, y_pred_dt)
f1_rf = f1_score(y_test, y_pred_rf)
f1_svm = f1_score(y_test, y_pred_svm)

roc_auc_lr = roc_auc_score(y_test, y_pred_lr)
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
roc_auc_rf = roc_auc_score(y_test, y_pred_rf)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)

# Print the performance of each model
print('Logistic Regression - Accuracy:', acc_lr, 'Precision:', prec_lr, 'Recall:', rec_lr, 'F1 Score:', f1_lr, 'ROC AUC:', roc_auc_lr)
print('Decision Tree - Accuracy:', acc_dt, 'Precision:', prec_dt, 'Recall:', rec_dt, 'F1 Score:', f1_dt, 'ROC AUC:', roc_auc_dt)
print('Random Forest - Accuracy:', acc_rf, 'Precision:', prec_rf, 'Recall:', rec_rf, 'F1 Score:', f1_rf, 'ROC AUC:', roc_auc_rf)
print('SVM - Accuracy:', acc_svm, 'Precision:', prec_svm, 'Recall:', rec_svm, 'F1 Score:', f1_svm, 'ROC AUC:', roc_auc_svm)

"""We can see that Decision Tree and Random Forest have highest accuray and F1 score so we will choose Random Forest model."""

import pickle
pickle_out = open("classifier.pkl",'wb')
pickle.dump(rf,pickle_out)
pickle_out.close()

y_pred_rf_new = rf.predict(test)
# Print the predictions
print('Random Forest - Predictions:', y_pred_rf_new)

'''from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Define the hyperparameters to tune
hyperparameters = {'C': [0.1, 1, 10, 100]}

# Create a Logistic Regression model object
model1 = LogisticRegression()

clf = GridSearchCV(model1, hyperparameters, cv=5)

# Train the model on the training data
clf.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = clf.predict(X_test)

# Evaluate the performance of the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))'''

'''from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

# Define the hyperparameters you want to tune
hyperparameters = {
    'max_depth': [3, 5, 7, None],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Create a Decision Tree model object
model2 = DecisionTreeClassifier()

# Create a GridSearchCV object
grid_search = GridSearchCV(estimator=model2, param_grid=hyperparameters, cv=5, n_jobs=-1)

# Train the model on the training data
grid_search.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = grid_search.predict(X_test)

# Evaluate the performance of the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
'''

'''from sklearn.ensemble import RandomForestClassifier

# Define the parameter grid to search over
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt', 'log2']
}


# Create a Random Forest model object
model3 = RandomForestClassifier()

grid_search1= GridSearchCV(
    estimator=model3,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)


# Train the model on the training data
grid_search1.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = grid_search1.predict(X_test)

# Evaluate the performance of the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))'''

'''from sklearn.svm import SVC

param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}

# Create a SVM model object
model4 = SVC()

grid = GridSearchCV(model4, param_grid, cv=5, scoring='accuracy')

# Train the model on the training data
grid.fit(X_train, y_train)

# Predict the target variable for the test data
y_pred = grid.predict(X_test)

# Evaluate the performance of the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))'''

