import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set()

data=pd.read_csv("1.01. Simple linear regression.csv")

y=data['GPA']
x=data['SAT']

x_matrix=x.values.reshape(-1,1)

reg=LinearRegression()


reg.fit(x_matrix,y)

#R-squared
print(reg.score(x_matrix,y))

#make a prediction
new_data=pd.DataFrame(data=[1740,1760],columns=['SAT'])
print(reg.predict(new_data))

plt.scatter(x,y)
yhat=reg.coef_*x_matrix+reg.intercept_
fig=plt.plot(x,yhat,lw=4,c='orange',label='regression line')
plt.xlabel('SAT',fontsize=20)
plt.ylabel('GPA',fontsize=20)
plt.show()
