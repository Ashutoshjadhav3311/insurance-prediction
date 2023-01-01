import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
medical = pd.read_csv('F:\college\Sem 5\Mini project\insurance.csv') # Read the given CSV file
#Mapping Male = 0; Female = 1 Non-somker = 0; Smoker = 1
medical['sex'] = medical['sex'].map({'male': 0, 'female': 1})
medical['smoker'] = medical['smoker'].map({'yes': 1, 'no': 0})
bins = [17,35,55,100] #Binning the age column.
slots = ['Young_adult','Senior_Adult','Elder']
medical['Age_range']=pd.cut(medical['age'],bins=bins,labels=slots)
medical.nunique().sort_values()
region=pd.get_dummies(medical.region,drop_first=True)
Age_range=pd.get_dummies(medical.Age_range,drop_first=True)
children= pd.get_dummies(medical.children,drop_first=True,prefix='children')
medical=pd.concat([region,Age_range,children,medical],axis=1)
#Drop region and age range as we are created a dummy
medical.drop(['region', 'Age_range', 'age','children'], axis = 1, inplace = True)
from sklearn.model_selection import train_test_split
medical_train, medical_test = train_test_split(medical, train_size = 0.7, random_state = 100)

#min - max scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
#Create a list of numeric variables
num_vars=['bmi','charges']
#Fit on data
medical_train[num_vars] = scaler.fit_transform(medical_train[num_vars])

#Divide the data into X and y
y_train = medical_train.pop('charges')
X_train = medical_train
#Recursive Feature Elimination
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 8
lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm, 8)             # running RFE
rfe = rfe.fit(X_train, y_train)
#List of variables selected
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]
X_train_rfe = X_train[col]
import statsmodels.api as sm  
X_train_new1 = X_train_rfe.drop(["children_5","children_4",'children_3'], axis = 1)
#Rebuilding the model
X_train_lm1 = sm.add_constant(X_train_new1)
lm1 = sm.OLS(y_train,X_train_lm1).fit()

#Drop the constant term B0
X_train_lm1 = X_train_lm1.drop(['const'], axis=1)

X_train_lm1=sm.add_constant(X_train_lm1)


y_train_pred = lm1.predict(X_train_lm1)


#Create a list of numeric variables
num_vars=num_vars=['bmi','charges']
#Fit on data
medical_test[num_vars] = scaler.transform(medical_test[num_vars])


y_test = medical_test.pop('charges')
X_test = medical_test
X_train_new1.columns

# Now let's use our model to make predictions.
# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new1.columns]
# Adding a constant variable 
X_test_new1 = sm.add_constant(X_test_new)


y_pred = lm1.predict(X_test_new1)
y_pred.head(20)
import streamlit as st



Const1=st.slider("select const",int(X_test_new1.const.min()),int(X_test_new1.const.max()))
Senior_Adult1 = st.number_input('Is senior adult? 1=Yes 0=No')
Elder1 = st.number_input('Is Elder? 1=Yes 0=No')
children_1 = st.number_input('Have 2 children? 1=Yes 0=No')   
bmi1 = st.slider("What is your bmi ?",float(X_test_new1.bmi.min()),float(X_test_new1.bmi.max()))
smoker1 = st.number_input('Are you smoker ?')       

predictions = lm1.predict([[Const1,Senior_Adult1,Elder1,children_1,bmi1,smoker1]])
if st.button("Prediction result"):
    st.write(predictions)                      