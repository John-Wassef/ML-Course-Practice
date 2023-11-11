import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import statsmodels.api as sm
import seaborn as sns

sns.set()

raw_train=pd.read_csv('2.02. Binary predictors.csv')
raw_test=pd.read_csv('2.03. Test dataset.csv')

train=raw_train.copy()
train['Admitted']=train['Admitted'].map({'Yes':1, 'No':0})
train['Gender']=train['Gender'].map({'Female':1, 'Male':0})

test=raw_test.copy()
test['Admitted']=test['Admitted'].map({'Yes':1, 'No':0})
test['Gender']=test['Gender'].map({'Female':1, 'Male':0})

y=train['Admitted']
x1=train[['Gender', 'SAT']]

x=sm.add_constant(x1)
reg_log=sm.Logit(y,x)
results_log=reg_log.fit()

test_actual=test['Admitted']
test_data=test.drop(['Admitted'],axis=1)
test_data=sm.add_constant(test_data)

def confusion_matrix(data,actual_values,model):

    pred_value=model.predict(data)
    bins=np.array([0,0.5,1])
    cm=np.histogram2d(actual_values,pred_value,bins=bins)[0]
    accuracy=(cm[0,0]+cm[1,1])/cm.sum()
    return cm,accuracy

cm=confusion_matrix(test_data, test_actual,results_log)
print(cm)