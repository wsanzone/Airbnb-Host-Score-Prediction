import numpy as np
import pandas as pd

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

listings = pd.read_csv(filepath1)

listings.drop(columns='Unnamed: 0', inplace=True)

listings.drop(columns=['picture_url', 'host_url'], inplace=True)

date_cols = ['last_scraped', 'host_since', 'calendar_last_scraped', 'first_review', 'last_review']
for item in date_cols:
    listings[item] = pd.to_datetime(listings[item])

listings.set_index('id', inplace=True)

cols_to_drop = ['listing_url', 'neighborhood_description', 'host_id', 'host_name', 'host_location',
                'host_neighbourhood']
listings.drop(columns=cols_to_drop, inplace=True)

clean_verifications_list = []
for item in listings['host_verifications']:
    string = item.replace('"', "")
    string2 = string.replace("'", '')
    string3 = string2.replace(",", "")
    string4 = string3.replace("[", "")
    string5 = string4.replace(']', '')
    clean_verifications_list.append(string5.split(" "))

listings['host_verifications'] = clean_verifications_list

exploded_verifications = listings['host_verifications'].explode().value_counts()[0::]
final_verifications = exploded_verifications[0:12]

top_verifications = list(final_verifications.index)

final_verifications_list = []
for row in clean_verifications_list: # looping through each item within our clean tags list
        final_verifications_list.append(list(set(row).intersection(top_verifications)))

listings['top_verifications'] = final_verifications_list
listings.drop(columns='host_verifications', inplace=True)

clean_amenities_list = []
for item in listings['amenities']:

    string = item.replace('"', "")
    string2 = string.replace("'", "")
    string4 = string2.replace("[", "")
    string5 = string4.replace(']', '')
    clean_amenities_list.append(string5.split(", "))

listings['amenities'] = clean_amenities_list

exploded_amenities = listings['amenities'].explode().value_counts()[0::]

final_amenities = exploded_amenities[0:15]

top_amenities = list(final_amenities.index)

final_amenities_list = []
for row in clean_amenities_list: # looping through each item within our clean tags list
        final_amenities_list.append(list(set(row).intersection(top_amenities)))

listings['amenities'] = final_amenities_list

listings['review_scores_rating'] = np.where(listings['review_scores_rating'] >= 4.7, 1, 0)

X = listings.drop(columns='review_scores_rating')
y = listings['review_scores_rating']

from sklearn.model_selection import train_test_split

X_train_rem, X_test, y_train_rem, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_rem, y_train_rem, test_size=0.3, random_state=42)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
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

name_transformer = Pipeline([
    ('name', TfidfVectorizer(strip_accents='unicode', stop_words='english', max_features=40))]
)

desc_transformer = Pipeline([
    ('desc', TfidfVectorizer(strip_accents='unicode', stop_words='english', max_features=40))]
)


class DateFormatter(TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xdate = X.apply(pd.to_datetime)
        return Xdate

class DateEncoder(TransformerMixin):

    def fit(self, X, y=None):
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
    ('formatter', DateFormatter()),
    ('encoder', DateEncoder())
])

dt_features = ['last_scraped', 'host_since', 'calendar_last_scraped', 'first_review', 'last_review']
date_encoder.fit(X_train[dt_features])
X_train_dates = date_encoder.transform(X_train[dt_features])

def explode_dates(df):
    dt_features = ['last_scraped', 'host_since', 'calendar_last_scraped', 'first_review', 'last_review']
    date_encoder.fit(df[dt_features])
    df_dates = date_encoder.transform(df[dt_features])
    return df_dates

X_val_dates = explode_dates(X_val)
X_test_dates = explode_dates(X_test)

def column_transform_df(df):
    numeric_columns = df.select_dtypes(['int', 'float'])
    numeric_columns = list(numeric_columns.columns)
    
    ohe_cols = ['neighborhood', 'borough', 'property_type', 'room_type']
    list_cols = ['amenities', 'top_verifications']
    name = 'name'
    desc = 'description'
    dates = ['last_scraped', 'host_since', 'calendar_last_scraped', 'first_review', 'last_review']

    
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
    df_array = transformed_df.toarray()
    final_df = pd.DataFrame(df_array, columns=columns, index=df.index)
    return final_df


X_train_ct = column_transform_df(X_train)
X_val_ct = column_transform_df(X_val)
X_test_ct = column_transform_df(X_test)

import functools as ft

train_dfs = [X_train_dates, X_train_num, X_train_ct]
val_dfs = [X_val_dates, X_val_num, X_val_ct]
test_dfs = [X_test_dates, X_test_num, X_test_ct]


X_train = ft.reduce(lambda left, right: pd.merge(left, right, on='id'), train_dfs)
X_val = ft.reduce(lambda left, right: pd.merge(left, right, on='id'), val_dfs)
X_test = ft.reduce(lambda left, right: pd.merge(left, right, on='id'), test_dfs)

for column in X_train.columns:
    if column not in X_test.columns:
        X_test[column] = 0

for column in X_test.columns:
    if column not in X_train.columns:
        X_test.drop([column], axis=1, inplace=True)


for column in X_train.columns:
    if column not in X_val.columns:
        X_val[column] = 0

for column in X_val.columns:
    if column not in X_train.columns:
        X_val.drop([column], axis=1, inplace=True)

if X_train.shape[1] != X_val.shape[1] != X_test.shape[1]:
    raise Exception('Oops! Your train, validation and test sets do not have the same amount of columns.')

