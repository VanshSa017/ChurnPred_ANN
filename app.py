import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

import streamlit as st

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.write("App Started...")



# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    sc = pickle.load(file)

with open('onehotencoder_geography.pkl','rb') as file:
    onehotencoder_geography = pickle.load(file)

st.title('Customer Churn Prediction')

# User Input
name = st.text_input('Customer Name')

geo_list = onehotencoder_geography.categories_[0]
geography = st.selectbox('Geography', options=geo_list)

gender_list = label_encoder_gender.classes_
gender = st.selectbox('Gender', options=gender_list)

age = st.number_input('Age', 18, 100)
balance = st.number_input('Balance', 0)
credit_score = st.number_input('Credit Score', 300, 850)
estimated_salary = st.number_input('Estimated Salary', 0)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', options=[0,1])
is_active_member = st.selectbox('Is Active Member', options=[0,1])

# Prepare input data
input_data = {
    'CreditScore': credit_score,
    'Gender': gender,
    'Geography': geography,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

input_data_df = pd.DataFrame([input_data])

# Encode Geography
geo_encoded = onehotencoder_geography.transform(input_data_df[['Geography']])
geo_encoded_df = pd.DataFrame(
    geo_encoded.toarray(),
    columns=onehotencoder_geography.get_feature_names_out()
)

# Encode Gender
input_data_df['Gender'] = label_encoder_gender.transform(input_data_df['Gender'])

# Combine data
input_data_df = pd.concat(
    [input_data_df.drop(columns=['Geography']), geo_encoded_df],
    axis=1
)

# Scale
input_data_scaled = sc.transform(input_data_df)

# Prediction
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

# Output
if prediction_prob >= 0.5:
    st.error(f'The Customer named {name} is likely to churn...')
else:
    st.success(f'The Customer named {name} is likely to stay...')