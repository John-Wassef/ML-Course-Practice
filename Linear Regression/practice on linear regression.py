import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

sns.set()

raw_data=pd.read_csv("../Data/1.04. Real-life example.csv")
data=raw_data.drop(['Model'],axis=1)
# print(data.describe(include="all"))
# print(data.isnull().sum())

#Clearing the data

data_no_mv=data.dropna(axis=0)

#Dropping incorrect data

q=data_no_mv['Price'].quantile(0.99)
data_1=data_no_mv[data_no_mv['Price']<q]

q=data_no_mv['Mileage'].quantile(0.99)
data_2=data_1[data_1['Mileage']<q]

data_3=data_2[data_2['EngineV']<6.5]

q=data_no_mv['Year'].quantile(0.01)
data_4=data_3[data_3['Year']>q]

data_clean=data_4.reset_index(drop=True)
# print(data_clean.describe(include='all'))

#log transformation to make the price linear
log_price=np.log(data_clean['Price'])
data_clean['Log Price']=log_price
data_clean=data_clean.drop(['Price'],axis=1)

#checking for multicorllinearity using variance inflation factor
# variables=data_clean[['Mileage','Year','EngineV']]
# vif=pd.DataFrame()
# vif['VIF']=[variance_inflation_factor(variables,i)for i in range(variables.shape[1])]
# vif['features']=variables.columns
# print(vif)

data_no_multicorllinearity=data_clean.drop(['Year'],axis=1)
data_with_dummies=pd.get_dummies(data_no_multicorllinearity,drop_first=True)

# print(data_with_dummies.columns.values)
cols=['Log Price','Mileage',
      'EngineV','Brand_BMW',
      'Brand_Mercedes-Benz','Brand_Mitsubishi',
      'Brand_Renault','Brand_Toyota',
      'Brand_Volkswagen','Body_hatch',
      'Body_other','Body_sedan',
      'Body_vagon','Body_van',
      'Engine Type_Gas','Engine Type_Petrol','Registration_yes']

data_preprossed=data_with_dummies[cols]

targets=data_preprossed['Log Price']
inputs=data_preprossed.drop(['Log Price'],axis=1)


scaler=StandardScaler()
scaler.fit(inputs)
inputs_scaled=scaler.transform(inputs)


x_train,x_test,y_train,y_test=train_test_split(inputs_scaled,targets,test_size=0.2,random_state=365)
y_test=y_test.reset_index(drop=True)

reg=LinearRegression()
reg.fit(x_train,y_train)

yhat=reg.predict(x_train)
# plt.scatter(yhat,y_train)
# plt.show()

#Rsquare
print(reg.score(x_train,y_train))

yhat_test=reg.predict(x_test)

df_pf=pd.DataFrame(np.exp(yhat_test),columns=['Prediction'])
df_pf['Target']=np.exp(y_test)
df_pf['Residual']=df_pf['Target']-df_pf['Prediction']
df_pf['Differance%']=np.abs(df_pf['Residual']/df_pf['Target']*100)
print(df_pf.describe())