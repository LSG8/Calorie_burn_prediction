# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:30:22 2021

@author: jhini
"""

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

exercise_df = pd.read_csv('exercise.csv')
#print(exercise_df.head())
calories_df = pd.read_csv('calories.csv')
df = pd.merge(exercise_df, calories_df, on = 'User_ID')
df = df[df['Calories'] < 300]
df = df.reset_index()
df['Intercept'] = 1
print(df.head())

df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)

print(df.corr()['Calories'])

#Plot code 1
# plt.figure(figsize=(8, 8))
# plt.plot(df['Duration'], df['Calories'], 'ro')
# plt.xlabel('Duration (min)', size = 18)
# plt.ylabel('Calories', size = 18) 
# plt.title('Calories burnt vs Duration of Exercise', size = 20)

#Plot code 2
# plt.scatter(df.Duration, df.Calories,  color='black')
# plt.xlabel("Duration (min)")
# plt.ylabel("Calories")
# plt.show()

#Plot code 3
# sns.regplot(x="Duration", y="Calories", data=df)
# plt.ylim(0,)

pearson_coef, p_value = stats.pearsonr(df['Gender'], df['Calories'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

x = np.asanyarray(df[['Age','Weight','Duration','Heart_Rate','Body_Temp']])
y = np.asanyarray(df[['Calories']])

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)

regr = linear_model.LinearRegression()

regr.fit (x_train, y_train)
# The coefficients
print ('Coefficients: ', regr.coef_)
print('R^2 value: ', regr.score(x_train,y_train))

yhat = regr.predict(x_test)
print("Residual sum of squares: %.2f" % np.mean((yhat - y_test) ** 2))


