# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:54:52 2019

@author: snarayanaswamy

Build a RandomForest model on TEMPOE dataset with an objective to compare performance against 
"""

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
pd.set_option('display.max_columns', None) ## Removes any restriction on default 
pd.set_option('display.max_rows', None) ## Removes any restriction on default 
import numpy as np
from sklearn.model_selection import train_test_split


## IMPORTING DATA
masterdata = pd.read_csv("C:/Users/snarayanaswamy/Downloads/multicoldataset_v2.csv", encoding='latin-1', low_memory=False)

## DATA EXPLORATION
masterdata.shape ## Data columns x rows Out[110]: (199972, 395)
masterdata.ndim ## number of dimension, mostly 2 dimension data
type(masterdata) ## type of the dataset - check if it is a dataframe or not
masterdata.head(10) ## Prints top 10 rows
masterdata.tail(10) ## Prints bottom 10 rows
masterdata.dtypes ##lists out all the variable names and datatypes

## VARIABLE DISTRIBUTION
masterdata.describe() ## mean,std, min, max, 25%, 50%,75% for all attributes
perc=[0.20,0.40,0.60,0.80]
include=['object','float','int']
masterdata["LN_Segment"].describe(percentiles=perc, include=include) ## mean,std, min, max, 25%, 50%,75% for only select attributes

## DROPPING A FEW VARIABLES FOR THIS ITERATION
masterdata = masterdata.drop(['Lease_storebkey_applied','DMANAME','PII_City','PII_Zip','PII_ApplicantID'],axis=1)
masterdata.shape ## Data columns x rows Out[110]: (199972, 395)

## CREATING DUMMIES FOR ALL CHARACTER PREDICTORS
masterdata=pd.get_dummies(masterdata) ## Creating dummies for all 
variable_type = pd.DataFrame(masterdata.dtypes) ##lists out all the variable names and datatypes
type(variable_type)
variable_type.shape
variable_type.head()

## WRITING OUTPUT TO A EXCEL FILE
writer = pd.ExcelWriter('C:/Users/snarayanaswamy/Downloads/testxls.xlsx', engine = 'xlsxwriter') ## Creating an excel file and assigning it to 'writer'
variable_type.to_excel(writer) ## Writing data from 'variable_type' onto 'writer'
writer.save() ## Save the 'writer' file

## NO FEATURE ENGINEERING WAS DONE FOR THIS ITERATION

## SPLITTING DATASET INTO DEPENDENT VARIABLE AND OTHER PREDICTORS

## CREATING Y DEPENDENT VARIABLE
y = masterdata[['Perf_2MP30']]
##Qn2: Is it required to convert pandas dataframes into numpy arrays before running randomforest?
y1 = np.array(masterdata['Perf_2MP30'])

## CREATING INDEPENDENT VARIABLES
x = masterdata.drop('Perf_2MP30', axis=1)
x1 = np.array(masterdata.drop('Perf_2MP30', axis=1))

##CREATING TRAINING & TESTING DATASET
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3)

y_train.shape
print('training predictors:',x_train.shape)
print('testing predictors:',x_test.shape)
print('training y:',y_train.shape)
print('testing y:',y_test.shape)

##TRAINING THE MODEL
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)

model = clf.fit(x_train,y_train.values.ravel()) ## check with Xiao, why would I get the error if I dont use ravel()?

##PREDICTING ON TESTING DATASET
y_pred = clf.predict(x_test)

## CHECKING ACCURACY OF ACTUAL VS. PREDICTED IN TESTING DATASET
from sklearn import metrics

print("Accuracy:", metrics.accuracy_score(y_test,y_pred))

##FEATURE IMPORTANCE
feature_imp = pd.Series(clf.feature_importances_, index =x_train.columns).sort_values(ascending=True) 
feature_imp






