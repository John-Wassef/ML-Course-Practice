import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

sns.set()

raw_data=pd.read_csv("1.03. Dummies.csv")

data=raw_data.copy()
data['Attendance']=data['Attendance'].map({'Yes':1,'No':0})

y=data['GPA']
x1=data[['SAT','Attendance']]

x=sm.add_constant(x1)
results=sm.OLS(y,x).fit()
# print(results.summary())

plt.scatter(data['SAT'],y,c=data['Attendance'],cmap='RdYlGn_r')

yhat_no=0.6439 + 0.0014 * data['SAT']
yhat_yes=0.8665 + 0.0014 * data['SAT']
yhat=0.0017*data['SAT']+0.275

# fig =plt.plot(data['SAT'],yhat_no,lw=2,c='red')
# fig =plt.plot(data['SAT'],yhat_yes,lw=2,c='green')
# fig =plt.plot(data['SAT'],yhat,lw=2,c='blue')
# plt.xlabel('SAT',fontsize=20)
# plt.xlabel('GPA',fontsize=20)
# plt.show()

new_data=pd.DataFrame({'const':1,'SAT':[1700,1670],'Attendance':[0,1]})
#to overwrite the dataframe arranging the colomes by alfapitical order
new_data=new_data[['const','SAT','Attendance']]
new_data.rename(index={0:'Bob',1:'Alice'},inplace=True)

predictions=results.predict(new_data)
predictionsf=pd.DataFrame({'Predictions':predictions})
joined=new_data.join(predictionsf)
joined.rename(index={0:'Bob',1:'Alice'},inplace=True)
print(joined)

