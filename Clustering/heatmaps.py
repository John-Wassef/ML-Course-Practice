import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

sns.set()

data=pd.read_csv('../Data/Country clusters standardized.csv',index_col='Country')
x_scaled=data.copy()
x_scaled=x_scaled.drop(['Language'],axis=1)


sns.clustermap(x_scaled,cmap='mako')
