import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler

sns.set()

data= pd.read_csv("1.02. Multiple linear regression.csv")

y=data['GPA']
x=data[['SAT','Rand 1,2,3']]

scaler=StandardScaler()
scaler.fit(x)
x_scaled=scaler.transform(x)


reg=LinearRegression()
reg.fit(x_scaled,y)

#making r2adj
r2=reg.score(x_scaled,y)
n=x.shape[0]
p=x.shape[1]
r2adj=1-(1-r2)*(n-1)/(n-p-1)
print(r2adj)

reg_summary=pd.DataFrame(data=x.columns.values,columns=['Features'])
reg_summary['Weights']=reg.coef_
reg_summary['P_Values']=f_regression(x,y)[1].round(3)
print(reg_summary)

new_data=pd.DataFrame(data=[[1700,2],[1800,1]],columns=['SAT','Rand 1,2,3'])
new_data_scaled=scaler.transform(new_data)
print(reg.predict(new_data_scaled))

