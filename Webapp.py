#All the data given should be in integer or float.

import numpy as np
import pickle
import streamlit as st
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv(".\diabetes.csv")
df=df.drop_duplicates()

df_selected=df.drop(['BloodPressure','Insulin','DiabetesPedigreeFunction','Outcome'],axis='columns')

df_selected['Glucose']=df_selected['Glucose'].replace(0,df_selected['Glucose'].mean())#normal distribution
df_selected['SkinThickness']=df_selected['SkinThickness'].replace(0,df_selected['SkinThickness'].median())#skewed distribution
df_selected['BMI']=df_selected['BMI'].replace(0,df_selected['BMI'].median())#skewed distribution


# loading the saved model
loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

st.snow()

def diabetes_prediction(input_data):
# creating a function for Prediction

    name_dict = {'Pregnancies':[input_data[0]],	'Glucose':[input_data[1]],	'SkinThickness':[input_data[2]],	'BMI':[input_data[3]],	'Age':[input_data[4]]}
    print("VALUE",name_dict)
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
      new_input_data = tuple(i)
      print("VALUE",new_input_data)

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(new_input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic.'
    else:
      return 'The person is diabetic.'
  
    
  
def main():
    
    # giving a title
    st.title('Diabetes Prediction Web App')
    st.text('All the data given should be in integer or float.')
    
    # getting the input data from the user   
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    SkinThickness = st.text_input('Skin Thickness value')
    BMI = st.text_input('BMI value')
    Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, SkinThickness, BMI, Age])
        
        
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()