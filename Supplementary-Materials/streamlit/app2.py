from gettext import npgettext
from turtle import pd, st


import pandas as pd
import streamlit as st
import joblib
import numpy as np
from datetime import date



st.title('Host Review Score Prediction for New York City')

@st.cache(allow_output_mutation=True)

listings = pd.read_csv('final_listings.csv')
X_train = pd.read_csv('X_train_processed.csv')

user_inputs = pd.DataFrame(columns=listings.columns)

user_inputs.drop(columns='Unnamed: 0', inplace=True)

# Putting the columns in a list to be looped over
date_cols = ['last_scraped', 'host_since', 'calendar_last_scraped', 'first_review', 'last_review']

# converting the datatype of each column in the for loop
for item in date_cols:
    user_inputs[item] = pd.to_datetime(user_inputs[item])

user_inputs.set_index('id', inplace=True)

# Putting our column names into a list
cols_to_drop = ['listing_url', 'neighborhood_description', 'host_id', 'host_name', 'host_location',
                'host_neighbourhood']

# Removing the columns from our dataframe
user_inputs.drop(columns=cols_to_drop, inplace=True)

##############################################################
#NOW WE CAN ACTUALLY TAKE USER INPUTS IN FIELDS
##############################################################

st.subheader("Please enter some information about your listing and we will predict if your review score will be good, or if it will need improvement")

