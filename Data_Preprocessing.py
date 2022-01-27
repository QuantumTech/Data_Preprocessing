
# This regression uses the absolute insensitive tube.
# The dots outside the absolute insensitive tube are the support vector regressions.
# Notes to self: Look into kernal SVR (3D graph.)
# Notes to self: SVM Kernal!

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Importing the libraries

# 'np' is the numpy shortcut!
# 'plt' is the matplotlib shortcut!
# 'pd' is the pandas shortcut!

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Importing the dataset

# Data set is creating the data frame of the 'Data.csv' file
# Features (independent variables) = The columns the predict the dependent variable
# Dependent variable = The last column
# 'X' = The matrix of features (country, age, salary)
# 'Y' = Dependent variable vector (purchased (last column))
# '.iloc' = locate indexes[rows, columns]
# ':' = all rows (all range)
# ':-1' = Take all the columns except the last one
# '.values' = taking all the values

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
print(X)
y = dataset.iloc[:, -1].values # NOTICE! .iloc[all the rows, only the last column]
print(y)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Taking care of missing data

# 'sklearn' = data science library
# Replacing the missing data with the average of the column it is found in.
# 'imputer' (object of the 'simpleImputer' class) = creating an instance of the class
# 'SimpleImputer(Which missing values?, replace with the mean)'
# '.fit' = connects 'imputer' to the matrix of features
# .fit(X[All the rows, Only the first and third columns (1:3)])

from sklearn.impute import SimpleImputer # Importing the library
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3]) # Where is the missing data located? (be aware of Strings)
X[:, 1:3] = imputer.transform(X[:, 1:3]) # Replacing the missing data values with the mean
print(X)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Encoding categorical data

# Encoding the Independent Variable

# What is happening here? = Turning the string columns (countries) into unique binary vectors

# 'One hot encoding' = Splitting a column up using the unique values. Creating binary vectors for each unique value
# 'ct' (object of the 'ColumnTransformer' class) = Creating an instance of the 'ColumnTransformer' class
# 'ColumnTransformer(transformers=[(The kind of transformation, What kind encoding, index of the columns we want to encode)], remainder = 'passthrough')'

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Data preprocessing

# Encoding categorical data

# Encoding the Dependent Variable

# Reminder! The dependent variable is the last column!
# Transforming the 'yes/no' from the last columns into binary values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Splitting the dataset into the Training set and Test set

# Note to self! = Split the data before feature scaling!
# Test set = future data
# Feature scaling = scaling the features so that they all take values in the same scale
# 80/20 split
# 'test_size' = 20% for the test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train) # The matrix of the features of the training set
print(X_test) # The matrix of the features of the test set
print(y_train) # The dependent variable of the training set
print(y_test) # The dependent variable of the test set

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Feature Scaling

# Why feature scale after splitting the data? = Because the 'test set' is a brand new set that is not supposed to be part of the training set

# This stage is not essential!

# Regression: simple linear regression, multiple linear regression, polynomial linear regression

# There are two feature scaling techniques STANDARDISATION & NORMALISATION

# Normalisation: All the values of the features will be between 0 and 1

# Normalisation: Recommended when you have a normal distribution in most of your features.

# Standardisation: All the values of the features will be between -3 and 3

# Standardisation: Works well all the time (RECOMMENDED)

# 'sc' = An object
# '3:' = Take all the columns to the from the third one onwards!

from sklearn.preprocessing import StandardScaler # Using standardisation
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
print(X_train)
print(X_test)

