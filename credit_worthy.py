import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('task1_dataset.csv')

df1 =  df.copy()

df1.drop(columns=['Age','Education','Gender','Residence_Type','Marital_Status'],inplace=True)

from sklearn.preprocessing import OrdinalEncoder,StandardScaler

oe = OrdinalEncoder()
ss = StandardScaler()

df1[['Payment_History','Employment_Status']] = oe.fit_transform(df1[['Payment_History','Employment_Status']])

df1[['Income','Debt']] = ss.fit_transform(df1[['Income','Debt']])

x = df1.drop(columns=['Creditworthiness'])
y = df1['Creditworthiness']

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(x_train,y_train)

y_pred = rfc.predict(x_test)

from sklearn.metrics import recall_score,f1_score,accuracy_score,roc_auc_score,precision_score

print(precision_score(y_test,y_pred))
print(recall_score(y_test,y_pred))
print(f1_score(y_test,y_pred))
print(roc_auc_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))

import joblib

joblib.dump(rfc ,'credit_worthy_predictor.pkl')