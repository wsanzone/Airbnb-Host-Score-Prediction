import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import warnings
warnings.filterwarnings("ignore")

filepath1 = ""
def callback():
    global filepath1
    filepath1 = filedialog.askopenfilename()
    root.destroy()
root = tk.Tk()
btn = tk.Button(root, text="Click to select .csv file", command=callback)
btn.pack()
tk.mainloop()
print("User selected", filepath1)

# Change file for each dataset that you want cleaned
#file = "raw_data/nyc_june2022.csv"
df = pd.read_csv(filepath1)


# Converting relevant values to datetimes
date_list = ['last_scraped', 'calendar_last_scraped', 'first_review', 'last_review']
for col in date_list:
    df[col] = pd.to_datetime(df[col])

# Mapping true/false values to binary values
tf_cols = ['host_is_superhost', 'host_has_profile_pic',
            'host_identity_verified', 'has_availability', 'instant_bookable']

for col in tf_cols:
    df[col] = df[col].map({'t': 1, 'f': 0})


# Dropping the unnecessary columns
remove_list = ['scrape_id', 'host_thumbnail_url', 'host_picture_url', 'host_listings_count', 'neighbourhood', 'bathrooms',
              'minimum_minimum_nights', 'maximum_minimum_nights', 'minimum_maximum_nights', 'minimum_maximum_nights', 
              'maximum_maximum_nights', 'calendar_updated', 'license', 'calculated_host_listings_count_entire_homes',
              'calculated_host_listings_count_private_rooms', 'calculated_host_listings_count_shared_rooms']
df = df.drop(columns = remove_list)

# Querying the original dataframe to only include rows that have values for the name column
df = df[df["name"].str.contains("NaN") == False]

# doing the same as above but now for the description column
df = df[df["description"].str.contains("NaN") == False]

# filling the host_name unknowns with "unknown"
df['host_name'] = df['host_name'].fillna('unknown')

# Removing the rows that have NaN values for host_since
df = df[df["host_since"].str.contains("NaN") == False]

# Filling host_location NaN values with host_neighbourhood values
df.host_location.fillna(df.host_neighbourhood, inplace=True)

if df['host_location'].isna().sum() != 0:
    # Filling remaining host_location values with the neighbourhood_cleansed values
    df.host_location.fillna(df.neighbourhood_cleansed, inplace=True)
#else:
    #return

# filling in bathrooms_text
df.bathrooms_text.fillna(1, inplace=True)

# Cleaning the instances of half-baths
df['bathrooms_text'] = df['bathrooms_text'].str.replace('Half-bath', '0.5')
df['bathrooms_text'] = df['bathrooms_text'].str.replace('Shared half-bath', '0.5')
df['bathrooms_text'] = df['bathrooms_text'].str.replace('Private half-bath', '0.5')

# Removing text and converting to numeric type and renaming
df['bathrooms_text'] = df['bathrooms_text'].str.replace('[a-z]+', '')
df['bathrooms_text'] = df['bathrooms_text'].astype(float)
df.rename({'bathrooms_text': 'num_bathrooms'}, inplace=True, axis=1)

# Conditional logic for if nulls remain
if df['num_bathrooms'].isna().sum() != 0:
    df['num_bathrooms'].fillna(1, inplace=True)
#else:
    #return

# Filling values with the values from neighbourhood_cleansed
df.host_neighbourhood.fillna(df.neighbourhood_cleansed, inplace=True)

df['host_has_bio'] = np.where(df['host_about'].isna() == True, 0, 1)
# Dropping the column
df.drop(columns='host_about', inplace=True)

# Replacing NaNs with "none" for the neighborhood description
df['neighborhood_overview'] = df['neighborhood_overview'].fillna('none')


# Converting our response/acceptance rates to numeric
cols = ['host_response_rate', 'host_acceptance_rate']

for col in cols:
    df[col] = df[col].str.replace('%', "")
    df[col] = df[col].astype('float')

non_null_responses = df[df['host_response_rate'].isna()==False]
if non_null_responses['host_response_rate'].median() > non_null_responses['host_response_rate'].mean():
    df['host_response_rate'] = df['host_response_rate'].fillna(df['host_response_rate'].mean())
else:
    df['host_response_rate'] = df['host_response_rate'].fillna(df['host_response_rate'].median())

non_null_acceptance = df[df['host_acceptance_rate'].isna()==False]
if non_null_acceptance['host_acceptance_rate'].median() > non_null_acceptance['host_acceptance_rate'].mean():
    df['host_acceptance_rate'] = df['host_acceptance_rate'].fillna(df['host_acceptance_rate'].mean())
else:
    df['host_acceptance_rate'] = df['host_acceptance_rate'].fillna(df['host_acceptance_rate'].median())

df.drop(columns ='host_response_time', inplace=True)
df['host_response_rate'] = df['host_response_rate'] / 100
df['host_acceptance_rate'] = df['host_acceptance_rate'] / 100

if df['bedrooms'].isna().sum() > df['beds'].isna().sum():
    df['bedrooms'] = df.groupby('beds')['bedrooms'].transform(lambda x: x.fillna(x.value_counts().index[0]))
    df = df.dropna(subset='beds')
else:
    df['beds'] = df.groupby('bedrooms')['beds'].transform(lambda x: x.fillna(x.value_counts().index[0]))
    df = df.dropna(subset='bedrooms')

# Dropping nulls from first_review and last_review
df = df.dropna(subset = ['first_review', 'last_review'])

df = df.dropna(subset = ['review_scores_accuracy', 'review_scores_checkin', 'review_scores_communication', 
                        'review_scores_location', 'review_scores_value'])


df['host_since'] = pd.to_datetime(df['host_since'])





df.rename({'neighbourhood_group_cleansed': 'neighborhood_group', 'neighbourhood_cleansed': 'neighborhood', 
          'neighborhood_overview': 'neighborhood_description', 
           'calculated_host_listing_count': 'host_total_listings'}, inplace=True, axis=1)

df['price'] = df['price'].str.replace('$', '')
df['price'] = df['price'].str.replace(',', '')
df['price'] = df['price'].astype(float)

df.drop(columns=['minimum_nights_avg_ntm', 'maximum_nights_avg_ntm'], inplace=True)

# If any column in df is empty, drop it
for col in df.columns:
    if df[col].isna().sum() == df.shape[0]:
        df.drop(col, axis=1, inplace=True)

# Raise an exception if nulls still exist in the data
if df.isna().sum().sum() != 0:
    print(df.isna().sum())
    raise Exception('Oops! There are still some nulls in your data')
#else:
    #return


def callbacksave():
    global filepath2
    filepath2 = filedialog.asksaveasfilename()
    root.destroy()
root = tk.Tk()
btn = tk.Button(root, text="Click to select destination folder to save your file", command=callbacksave)
btn.pack()
tk.mainloop()
print("User selected", filepath2)
df.to_csv(filepath2)
print("Success! Your clean file has been sent to the folder you specified")

