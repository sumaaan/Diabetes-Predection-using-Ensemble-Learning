#pip install -U scikit-learn scipy matplotlib
#pip install numpy  

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv(".\diabetes.csv")
df=df.drop_duplicates()

df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())#normal distribution
df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].median())#skewed distribution
df['BMI']=df['BMI'].replace(0,df['BMI'].median())#skewed distribution

df_selected=df.drop(['BloodPressure','Insulin','DiabetesPedigreeFunction',],axis='columns')

target_name='Outcome'
y= df_selected[target_name]#given predictions - training data 
X= df_selected.drop(target_name,axis=1)#dropping the Outcome column and keeping all other columns as X

name_dict = {'Pregnancies':['6'],	'Glucose':['148'],	'SkinThickness':['35'],	'BMI':['33.6'],	'Age':['50']}
new_data_df=pd.DataFrame(name_dict)
new_df = pd.concat([df_selected, new_data_df], axis = 0, join ='inner')


x=new_df
quantile  = QuantileTransformer()
X = quantile.fit_transform(x)
df_new=quantile.transform(X)
df_new=pd.DataFrame(X)
df_new.columns =['Pregnancies', 'Glucose','SkinThickness','BMI','Age']


alist = (df_new.tail(1)).values.tolist()
for i in alist:
  print(i)
  input_data = tuple(i)


# loading the saved model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
print("INP data numpy", input_data_as_numpy_array)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
print("input data reshaped", input_data_reshaped)

prediction = loaded_model.predict(input_data_reshaped)
#prediction by voting classifier model.
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
