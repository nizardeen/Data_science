# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

os.chdir('C:\\Users\\D123005\\PycharmProjects\\Cars')
print(os.getcwd())

df_train = pd.read_excel('Data_Train.xlsx')
df_test = pd.read_excel('Data_Test.xlsx')

print(df_train.shape)
print(df_test.shape)

df = pd.concat([df_train,df_test],axis=0,sort=False)

df.shape
df.isnull().sum()

df.columns = [a.lower() for a in df.columns]

df.mileage.mode()
df.mileage.fillna(df.mileage.mode()[0],inplace=True)

colours = ['#000099', '#ffff00'] # specify the colours - yellow is missing. blue is not missing.
sns.heatmap(df.isnull(), cmap=sns.color_palette(colours))

for col in df.columns:
    pct_missing = np.mean(df.isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))

df.dropna(subset = ['engine','power'],axis=0,inplace=True)

df.drop('new_price',axis=1,inplace=True)

df.isnull().sum()

df.dropna(subset=['seats'],axis=0,inplace=True)

def cleanmileage(string):
    str_split = string.split()
    return float(str_split[0])

df.mileage = df.mileage.apply(cleanmileage,convert_dtype=True)

df.engine = df.engine.apply(cleanmileage,convert_dtype=True)
# df.power = df.power.apply(cleanmileage,convert_dtype=True)








