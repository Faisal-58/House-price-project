from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np


def preprocess_data(df):
    df.rename(columns={
        'Lattitude': 'Latitude',
        'Longtitude': 'Longitude'
    }, inplace=True)
    df.drop(columns=['Address','Date','YearBuilt'], inplace=True)
    X = df.drop('Price', axis=1)
    y = df['Price']
    y = np.log1p(y)


    

# Define preprocessing pipelines
    num_features = ['Rooms', 'Distance','Postcode','Bedroom2','Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Latitude','Longitude','Propertycount']
    cat_features = ['Suburb','Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname']

    num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

# Combine into a ColumnTransformer
    preprocessor = ColumnTransformer([
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train,y_test, preprocessor
