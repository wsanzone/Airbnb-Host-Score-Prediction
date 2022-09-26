from gettext import npgettext
from turtle import pd, st


import pandas as pd
import streamlit as st
import joblib
import numpy as np
from datetime import date



st.title('Host Review Score Prediction for New York City')

@st.cache(allow_output_mutation=True)
def load_data(path, num_rows):

    df = pd.read_csv(path, nrows=num_rows)

    return df
X_test = load_data('X_test_processed.csv', 30000)
#cols_to_remove = ['review_scores_accuracy', 'review_scores_cleanliness',
# 'review_scores_checkin', 'review_scores_communication',
 #'review_scores_location', 'review_scores_value']
#X_train = X_train.drop(columns=cols_to_remove)
listings = load_data("final_listings.csv", 30000)
listings = listings.drop(columns=["Unnamed: 0", 'geometry'])
#listings = listings.drop(columns='listing_url')

model = joblib.load('gradboost.pkl')
#st.map(listings)

st.subheader("Please enter some information about your listing and we will predict if your review score will be good, or if it will need improvement")
st.dataframe(listings)
picture_url = "picurl"
host_url = "hosturl"

review_scores_rating = 0

review_scores_accuracy = 0

review_scores_cleanliness = 0

review_scores_checkin = 0

review_scores_communication = 0

review_scores_location = 0

review_scores_value = 0


id = np.random.randint(1000, 10000, size=1)[0]
st.write(id)
listing_url = "listingurl"
st.write(listing_url)


last_scraped = "2022-06-04"
st.write(last_scraped)


name = st.text_input('Please enter the name of your listing:')
st.write(name)

description = st.text_input("Please enter a description for your listing:")
st.write(description)

neighborhood_description = "none"

host_id = 1234

host_name = "Tim"

host_since = "2022-09-25"

host_location = 'hostlocation'

host_response_rate = listings['host_response_rate'].mean()
st.write(host_response_rate)

host_acceptance_rate = listings['host_acceptance_rate'].mean()

host_is_superhost = 0

host_neighbourhood = "hostneighborhood"


neighborhood_group = st.selectbox('What Borough is your listing in?', listings['neighborhood_group'].unique())
st.write(neighborhood_group)


if neighborhood_group == "Queens":
    neighborhood = st.selectbox('What neighborhood is your listing in?', listings[listings['neighborhood_group']=="Queens"]['neighborhood'].unique())
elif neighborhood_group == "Brooklyn":
    neighborhood = st.selectbox('What neighborhood is your listing in?', listings[listings['neighborhood_group']=="Brooklyn"]['neighborhood'].unique())
elif neighborhood_group == "Manhattan":
    neighborhood = st.selectbox('What neighborhood is your listing in?', listings[listings['neighborhood_group']=="Manhattan"]['neighborhood'].unique())
elif neighborhood_group == "Bronx":
    neighborhood = st.selectbox('What neighborhood is your listing in?', listings[listings['neighborhood_group']=="Bronx"]['neighborhood'].unique())
elif neighborhood_group == "Staten Island":
    neighborhood = st.selectbox('What neighborhood is your listing in?', listings[listings['neighborhood_group']=="Staten Island"]['neighborhood'].unique())
else:
    pass
st.write(neighborhood)

host_total_listings_count = 1.0

host_verifications = st.multiselect('What ways can you verify your identity to your guests?', ['phone',
 'email',
 'government_id',
 'work_email',
 'reviews',
 'offline_government_id',
 'jumio',
 'selfie',
 'identity_manual',
 'kba',
 'facebook',
 'google'])


st.write(host_verifications)

host_has_profile_pic = 1

host_identity_verified = 1

latitude = 40.7831

longitude = -73.9712

property_type = st.selectbox("What kind of property type is your listing?", ('Private Room', 'Entire Apartment', 'Entire Condo', 'Entire Home', "Entire Townhome", "Entire Loft",
                            "Hotel Room", "Vacation Home", ))
st.write(property_type)


if property_type == "Private Room":
    room_type = "Private room"
else:
    room_type = st.selectbox('What type of room(s) does your listing offer?', ('Entire home/apt', 'Shared room', "Hotel room"))
st.write(room_type)

accommodates = st.slider("How many people can your place accommodate?", 1, 16)
st.write(accommodates)

num_bathrooms = st.slider("How many bathrooms does your listing have?", min_value=0.5, max_value=15.5, step=0.5)
st.write(num_bathrooms)

bedrooms = st.slider("How many bedrooms does your listing have?", 1, 15)
st.write(bedrooms)

beds = st.slider("How many beds does your listing have?", 1, 42)
st.write(beds)

amenities = st.multiselect("What amenities does your listing offer?", ('Wifi',
 'Smoke alarm',
 'Essentials',
 'Long term stays allowed',
 'Kitchen',
 'Heating',
 'Air conditioning',
 'Carbon monoxide alarm',
 'Hangers',
 'Hair dryer',
 'Hot water',
 'Iron',
 'Shampoo',
 'Dishes and silverware',
 'Refrigerator'))
st.write(amenities)


price = st.number_input('Please input the price per night you wish to charge for your listing:', min_value=0)
st.write(price)

minimum_nights = 1

maximum_nights = st.number_input('Please input the maximum number of nights you would allow someone to stay:', min_value=1)
st.write(maximum_nights)

has_availability = 1

availability_30 = listings['availability_30'].mean()

availability_60 = listings['availability_60'].mean()

availability_90 = listings['availability_90'].mean()

availability_365 = listings['availability_365'].mean()
st.write(availability_365)

calendar_last_scraped = '2022-06-04'

number_of_reviews = 0

number_of_reviews_ltm = 0

number_of_reviews_l30d = 0

first_review = '2022-06-04'

last_review = '2022-06-04'

instant_bookable = st.multiselect('Will you allow your listing to be instantely bookable?', ('Yes', "No"))

if instant_bookable == "Yes":
    instant_bookable = 1
else:
    instant_bookable = 0
st.write(instant_bookable)

calculated_host_listings_count = 1
reviews_per_month = 0.0
host_has_bio = 1

#################################################################################################################################
picture_url, host_url
# Encoding below
user_list = [id, listing_url, last_scraped, name, description, neighborhood_description, host_id, host_name, host_since, host_location, host_response_rate,
            host_acceptance_rate, host_is_superhost, host_neighbourhood, host_total_listings_count, host_verifications, host_has_profile_pic, host_identity_verified,
            neighborhood, neighborhood_group, latitude, longitude, property_type, room_type, accommodates, num_bathrooms, bedrooms, beds, amenities, price, minimum_nights,
            maximum_nights, has_availability, availability_30, availability_60, availability_90, availability_365, calendar_last_scraped, number_of_reviews,
            number_of_reviews_ltm, number_of_reviews_l30d, first_review, last_review, review_scores_rating, review_scores_accuracy, review_scores_cleanliness, 
            review_scores_checkin, review_scores_communication, review_scores_location, review_scores_value, instant_bookable, calculated_host_listings_count, reviews_per_month, host_has_bio]


#for col in listings.columns:
    #if col not in user_list:
        #user_list.append(col)

st.write(user_list)
user_df = pd.DataFrame([user_list], columns=listings.columns)


# Putting the columns in a list to be looped over
date_cols = ['last_scraped', 'host_since', 'calendar_last_scraped', 'first_review', 'last_review']

# converting the datatype of each column in the for loop
for item in date_cols:
    user_df[item] = pd.to_datetime(user_df[item])

user_df.set_index('id', inplace=True)

# Putting our column names into a list
cols_to_drop = ['listing_url', 'neighborhood_description', 'host_id', 'host_name', 'host_location',
                'host_neighbourhood']

# Removing the columns from our dataframe
user_df.drop(columns=cols_to_drop, inplace=True)


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.base import BaseEstimator, TransformerMixin

class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Wraps `MultiLabelBinarizer` in a form that can work with `ColumnTransformer`. Note
    that input X has to be a `pandas.DataFrame`.

    Requires the non-training DataFrame to ensure it collects all labels so it won't be lost in train-test-split

    To initialize, you musth pass the full DataFrame and not 
    the df_train or df_test to guarantee that you captured all categories.
    Otherwise, you'll receive a user error with regards to missing/unknown categories.
    """
    def __init__(self, df:pd.DataFrame):
        self.mlbs = list()
        self.n_columns = 0
        self.categories_ = self.classes_ = list()
        self.df = df
    
    def fit(self, X:pd.DataFrame, y=None):
        
        # Collect columns
        self.columns = X.columns.to_list()

        # Loop through columns
        for i in range(X.shape[1]): # X can be of multiple columns
            mlb = MultiLabelBinarizer()
            mlb.fit(self.df[self.columns].iloc[:,i])
            self.mlbs.append(mlb)
            self.classes_.append(mlb.classes_)
            self.n_columns += 1

        self.categories_ = self.classes_

        return self

    def transform(self, X:pd.DataFrame):
        if self.n_columns == 0:
            raise ValueError('Please fit the transformer first.')
        if self.n_columns != X.shape[1]:
            raise ValueError(f'The fit transformer deals with {self.n_columns} columns '
                             f'while the input has {X.shape[1]}.'
                            )
        result = list()
        for i in range(self.n_columns):
            result.append(self.mlbs[i].transform(X.iloc[:,i]))

        result = np.concatenate(result, axis=1)
        return result

    def fit_transform(self, X:pd.DataFrame, y=None):
        return self.fit(X).transform(X)

    def get_feature_names_out(self, input_features=None):
        cats = self.categories_
        if input_features is None:
            input_features = self.columns
        elif len(input_features) != len(self.categories_):
            raise ValueError(
                "input_features should have length equal to number of "
                "features ({}), got {}".format(len(self.categories_),
                                               len(input_features)))

        feature_names = []
        for i in range(len(cats)):
            names = [input_features[i] + "_" + str(t) for t in cats[i]]
            feature_names.extend(names)

        return np.asarray(feature_names, dtype=object)


# Each entry in the pipeline must be a tuple

name_transformer = Pipeline([ #Creating a pipeline to transform the name column
    ('name', TfidfVectorizer(strip_accents='unicode', stop_words='english', max_features=40))]
)

desc_transformer = Pipeline([ # Creating a pipeline to transform the description column
    ('desc', TfidfVectorizer(strip_accents='unicode', stop_words='english', max_features=40))]
)


class DateFormatter(TransformerMixin): # Formats the date into a datetime format

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xdate = X.apply(pd.to_datetime) # converts to datetime datatype
        return Xdate

class DateEncoder(TransformerMixin):
    """
    Takes in a column and converts it to that column's year, month and day
    """
    def fit(self, X, y=None): #stateless transformer
        return self

    def transform(self, X):
        dfs = []
        self.column_names = []
        for column in X:
            dt = X[column].dt
            # Assign custom column names
            newcolumnnames = [column+'_'+col for col in ['year', 'month', 'day']]
            df_dt = pd.concat([dt.year, dt.month, dt.day], axis=1)
            # Append DF to list to assemble list of DFs
            dfs.append(df_dt)
            # Append single DF's column names to blank list
            self.column_names.append(newcolumnnames)
        # Horizontally concatenate list of DFs
        dfs_dt = pd.concat(dfs, axis=1)
        return dfs_dt

    def get_feature_names(self):
        # Flatten list of column names
        self.column_names = [c for sublist in self.column_names for c in sublist]
        return self.column_names

date_encoder = Pipeline([
    ('formatter', DateFormatter()), #formatting portion of the pipeline
    ('encoder', DateEncoder()) #encoding portion of the pipeline
])

# Creating a list of all our datetime column names
dt_features = ['last_scraped', 'host_since', 'calendar_last_scraped', 'first_review', 'last_review']
date_encoder.fit(user_df[dt_features])
user_df_dates = date_encoder.transform(user_df[dt_features])

user_df_num = user_df.select_dtypes(['int', 'float'])

def column_transform_df(df):
    
    """
    INPUTS: Takes in a dataframe (will only work with notebooks that have been processed in the same ways as 
    this notebook)
    
    OUTPUTS: Returns a dataframe with the following:
    - One hot encoded columns
    - multilabel binarized columns
    - text columns that are transformed using the TfidfVectorizer
    """
    
    numeric_columns = df.select_dtypes(['int', 'float']) # Selecting numeric columns from the input df
    numeric_columns = list(numeric_columns.columns)
    
    ohe_cols = ['neighborhood', 'neighborhood_group', 'property_type', 'room_type'] # columns to be onehotencoded
    list_cols = ['amenities', 'host_verifications'] # columns to be multi-label binarized
    name = 'name' # Will be transformed using Tfidf
    desc = 'description' # Also transformed using Tfidf
    #dates = ['last_scraped', 'host_since', 'calendar_last_scraped', 'first_review', 'last_review']

    
    ct = ColumnTransformer(
    transformers=[#('scaler', StandardScaler(), num_cols),
                    #('dt', date_encoder, dates),
                 ('ohe', OneHotEncoder(), ohe_cols),
                 ('mhe', MultiHotEncoder(df), list_cols),
                  ('name', name_transformer, name),
                  ('desc', desc_transformer, desc)
                 ], remainder='drop'
    )
    
    
    ct.fit(df)
    columns = ct.get_feature_names_out()
    transformed_df = ct.transform(df)
    #df_array = transformed_df.toarray()
    final_df = pd.DataFrame(transformed_df, columns=columns, index=df.index)
    return final_df

user_df_ct = column_transform_df(user_df)

import functools as ft

user_dfs = [user_df_dates, user_df_num, user_df_ct]

user_df = ft.reduce(lambda left, right: pd.merge(left, right, on='id'), user_dfs)


for column in X_test.columns:
    if column not in user_df.columns: # If the column in X_train is not in X_test add it to X_test and set to zero
        user_df[column] = 0

for column in user_df.columns: # Checks if a column in X_test is not in X_train and drops it from X_test if true
    if column not in X_test.columns:
        user_df.drop([column], axis=1, inplace=True)

cols_to_remove = ['review_scores_accuracy', 'review_scores_cleanliness',
 'review_scores_checkin', 'review_scores_communication',
 'review_scores_location', 'review_scores_value']

user_df.drop(columns=cols_to_remove, inplace=True)



prediction = model.predict(user_df)

#if prediction == 0:
    #st.write("Your score could use some improvement")
#if prediction == 1:
    #st.write('You should have a good review score!')