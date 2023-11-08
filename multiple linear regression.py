import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

sns.set()

data= pd.read_csv("1.02. Multiple linear regression.csv")

y=data['GPA']
x1=data[['SAT','Rand 1,2,3']]

x=sm.add_constant(x1)
results=sm.OLS(y,x).fit()
print(results.summary())