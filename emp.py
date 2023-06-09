import pickle
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# loading the saved model
emp_ret_model = pickle.load(open('emp_retention_model.sav','rb'))

def emp_retention_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = emp_ret_model.predict(input_data_reshaped)
    if prediction[0] == 0:
        return "The Employee will LEAVE the company"
    else:
        return "The Employee will STAY in the company"
    

def main():
    st.title('Employee Retention Prediction Web Application')

    feature_explanations = {
        'Department': '7: Sales, 6: R&D, 5: Procurement, 4: Operations, 3: Legal, 2: HR, 1: Finance, 0: Analytics',
        'Education Level': '2: Master and above, 1: Bachelor, 0: Below Secondary',
        'Gender': '1: Male, 0: Female'
    }

    # Getting input data from user
    department = st.number_input('Department', min_value=0, max_value=7, step=1, value=0, help=feature_explanations['Department'])
    education = st.number_input('Education Level', min_value=0, max_value=2, step=1,value=0, help=feature_explanations['Education Level'])
    gender = st.number_input('Gender', min_value=0, max_value=1, step=1,value=0,help=feature_explanations['Gender'])
    trainings = st.number_input('Number of trainings attended',min_value=1, max_value=10, step=1,value=1)
    age = st.number_input('Age', min_value=18, max_value=60, step=1,value=18)
    previous_year_rating = st.number_input('Previous year rating', min_value=1, max_value=5, step=1,value=1)
    length_of_service = st.number_input('Length of service', min_value=1, max_value=20, step=1,value=1)
    awards_won = st.number_input('Awards won', min_value=0, max_value=1, step=1,value=0)
    avg_training_score = st.number_input('Average training score', min_value=39, max_value=99, step=1,value=39)
    #code for prediction
    result=''



    #creating button for prediction
    if st.button("Predict"):
        result = emp_retention_prediction([department, education, gender, trainings, age, previous_year_rating,length_of_service,awards_won,avg_training_score])

       


    st.success(result)
    

if __name__ == '__main__':
    main()


    






