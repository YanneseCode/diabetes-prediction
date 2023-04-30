import streamlit as st
import pickle
import pandas as pd


def main():
    style = """<div style='background-color:pink; padding:12px'>
              <h1 style='color:black'>Diabetes Prediction App</h1>
       </div>"""
    st.markdown(style, unsafe_allow_html=True)
    left, right = st.columns((2,2))
    Pregnancies = left.number_input('Enter pregnancy count of the patient', step =1.0,format="%.1f", value=0.0)
    Glucose = right.number_input('Enter glucose level of the patient',  step=1.0, format='%.2f', value= 80.00)
    BloodPressure = left.number_input('Enter your blood pressure (mmHg)', step=1.0, format='%.2f', value=66.00)
    SkinThickness = right.number_input('Enter the Skin Thickness of the patient (mm)', step=1.0, format='%.2f', value=23.00)
    Insulin = left.number_input('Enter insulin level of the patient', step=1.0, format='%.1f', value=0.0)
    BMI = right.number_input('Enter BMI score of the patient', step=1.0, format='%.1f', value=28.0)
    DiabetesPedigreeFunction = left.number_input('Enter your Diabetes Pedigree rate for the patient', step=1.0, format='%.3f', value=0.160)    
    Age = right.number_input('What is the current age for the patient',  step=1.0, format='%.1f', value=43.0)
    button = st.button('Predict')
    # if button is pressed
    if button:
        # make prediction
        result = predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                         BMI, DiabetesPedigreeFunction, Age)
        st.success(f'The diabetic outcome for the patient is ${result}')





# load the train model
with open('diabetesModel1', 'rb') as rf:
    model = pickle.load(rf)

# load the StandardScaler
with open('scalers', 'rb') as stds:
    scaler = pickle.load(stds)

def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                         BMI, DiabetesPedigreeFunction, Age):
    # processing user input
    lists = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                         BMI, DiabetesPedigreeFunction, Age]
    df = pd.DataFrame(lists).transpose()
    # scaling the data
    scaler.transform(df)
    # making predictions using the train model
    prediction = model.predict(df)
    result = int(prediction)
    return result



if __name__ == '__main__':
    main()
