import streamlit as st
import requests




st.set_page_config(page_title="Melbourne House Price Predictor")
st.title("🏡 Melbourne House Price Prediction")

st.markdown("Enter house details below:")
import pandas as pd
@st.cache_data
def load_raw_data():
    df = pd.read_csv("data/raw/aus.csv")
    df.rename(columns={'Lattitude': 'Latitude', 'Longtitude': 'Longitude'}, inplace=True)
    df.drop(columns=['Address', 'Date', 'YearBuilt'], inplace=True)
    return df

df = load_raw_data()

# Numeric inputs
rooms = st.number_input("Rooms", min_value=1, max_value=10, value=3)
distance = st.number_input("Distance from CBD (km)", min_value=0.0, max_value=50.0, value=5.0)
postcode = st.number_input("Postcode", min_value=3000, max_value=4000, value=3101)
bedroom2 = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
bathroom = st.number_input("Bathrooms", min_value=0, max_value=5, value=2)
car = st.number_input("Car Spaces", min_value=0, max_value=5, value=1)
landsize = st.number_input("Land Size (sqm)", min_value=0, max_value=1000, value=300)
building_area = st.number_input("Building Area (sqm)", min_value=0, max_value=1000, value=150)
latitude = st.number_input("Latitude", value=-37.81)
longitude = st.number_input("Longitude", value=144.96)
propertycount = st.number_input("Property Count", min_value=0, max_value=10000, value=5000)

# Categorical inputs
suburb = st.selectbox("Suburb", sorted(df['Suburb'].dropna().unique()))
type_ = st.selectbox("Type", ["h", "u", "t"]) 
method = st.selectbox("Method", ["S", "SP", "PI", "VB", "SA"])
sellerg = st.selectbox("SellerG", sorted(df['SellerG'].dropna().unique()))
council_area = st.selectbox("Council Area", sorted(df['CouncilArea'].dropna().unique()))
region = st.selectbox("Region Name", sorted(df['Regionname'].dropna().unique()))


# Prediction button
if st.button("🔍 Predict Price"):
    payload = {
        "Rooms": rooms,
        "Distance": distance,
        "Postcode": postcode,
        "Bedroom2": bedroom2,
        "Bathroom": bathroom,
        "Car": car,
        "Landsize": landsize,
        "BuildingArea": building_area,
        "Latitude": latitude,
        "Longitude": longitude,
       
        "Propertycount": propertycount,
        "Suburb": suburb,
        "Type": type_,
        "Method": method,
        "SellerG": sellerg,
        "CouncilArea": council_area,
        "Regionname": region
    }

    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        if response.status_code == 200:
            prediction = response.json()["predicted_price"]
            st.success(f"💰 Estimated Price: ${prediction:,.2f}")
        else:
            st.error("Failed to get prediction from API.")
    except Exception as e:
        st.error(f"Error: {e}")
