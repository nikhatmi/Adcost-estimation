#Base Libraries
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

#Trained Model Files
import joblib
import pickle
#import Saved model object files
ohe = joblib.load('ohe.pkl')
model = joblib.load("cost_rf.pkl")

#Loading saved onehotencoded files
with open("ohe.pkl", "rb") as f:  

    contentType_encoding = pickle.load(f)

#############Sample Input Data to Show to the User########################
data  = pd.read_excel('FINAL_DATASET_PREDICTIVE_MODELING_1.xlsx')   


###################Design of User Interface###############################

#Using streamlit python module

import streamlit as st

#Title
st.title("Cost Estimation of Ad campaign on Instagram using given data... ")

st.image("https://www.socialpilot.co/wp-content/uploads/2021/06/How-To-Run-Instagram-Ads-A-Beginners-Guide-To-Becoming-Pro.jpg")

st.subheader("Input Data Trained On")
st.dataframe(data.head())


######################################Taking User Input Values#######################################

col1,col2,col3,col4 = st.columns(4)

with col1:
    clicks= st.number_input("Enter no.of Clicks")
with col2: 
    impressions= st.number_input("Enter no.of Impressions")
with col3: 
    ctr= st.number_input("Enter no.of CTR(%)") 
with col4: 
    conversions= st.number_input("Enter no.of Conversions")          

col5,col6 = st.columns(2)
with col5:
    contenttype =  st.selectbox("Select Content type:", data['Content Type'].str.lower().unique())
with col6:
    engagementlevel =  st.selectbox("Select Engagement level:", data['EngagementLevel '].str.lower().unique())

if st.button("Estimate"):

    
    row = pd.DataFrame([[clicks, impressions, ctr, conversions,engagementlevel,contenttype]],
          columns=['Clicks','Impressions','CTR(%)','Conversions','EngagementLevel','Content Type']) 
    
    
    st.write('\nGiven Input data:')     

    st.dataframe(row)
   

    # Applying Feature Modification steps before giving it to model 
    #Ordinaly encoding
    row['EngagementLevel'].replace({'low':0, 'medium':1, 'high':2}, inplace=True)

    # Onehot Encoding
    row_ohe = ohe.transform(row[['Content Type']]).toarray()
    row_ohe = pd.DataFrame(row_ohe, columns=ohe.get_feature_names_out())
    
    row = pd.concat([row.iloc[:, 0:5], row_ohe], axis=1)


    Cost = round(model.predict(row)[0],2)
    
    st.write(f"Estimated AdCost Value: {Cost}")

