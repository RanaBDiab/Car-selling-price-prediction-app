import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler



# building interface 
st.title('Car selling price prediction app')
st.image('car_app_vec.jpg')
st.text('This app can help you predict the selling price for your used car')
st.text('please fill in the following data')

make = st.selectbox('Make', ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault','Mahindra', 'Tata', 'Chevrolet', 'Datsun','Volkswagen', 'Nissan', 'Fiat'])
model = st.selectbox('Model', ['Swift', 'Rapid', 'City', 'i20', 'Xcent', 'Wagon', '800', 'Etios', 'Figo', 'Duster', 'Zen', 'KUV', 'Ertiga', 'Alto', 'Verito', 'WR-V', 'SX4','Tigor', 'Baleno', 'Enjoy', 'Omni', 'Vitara', 'verna', 'GO', 'Safari', 'Innova','Amaze', 'Ciaz', 'jazz', 'Manza', 'i10', 'Ameo', 'Indica', 'Vento', 'EcoSport', 'Celerio','Polo', 'Eeco', 'Scorpio', 'Freestyle', 'Indigo', 'Corolla', 'Terrano', 'Creta', 'KWID','Santro', 'Elantra', 'Nexon', 'Ritz', 'Grand', 'Zest', 'Getz', 'Elite', 'Brio', 'Sunny','Micra', 'XUV500', 'Accent', 'Ignis', 'Tiago', 'Thar', 'Sumo', 'New', 'Bolero', 'Beat','A-Star', 'Nano', 'EON', 'RediGO', 'Fiesta', 'Civic', 'Sail', 'Ecosport', 'TUV', 'Xylo','Grande', 'S-Cross', 'Tavera', 'Linea', 'Esteem', 'Octavia', 'Spark', 'Optra', 'Mobilio', 'Cruze'] )
year = st.slider('Model year', 1991, 2020)
km_driven = st.number_input('Kilometers driven')
transmission = st.selectbox('Tranmission type', ['Manual', 'Automatic'])
fuel = st.selectbox('Fuel type', ['Diesel', 'Petrol', 'LPG', 'CNG'])
mileage = st.number_input('Mileage')
engine = st.number_input ('Engine CC')
max_power= st.number_input('Maximum power')
Nm = st.number_input('Neuton meter')
rpm = st.number_input('revolution per minute (rpm)')
seats = st.number_input('number of seats')
owner = st.selectbox('Owner', ['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner'])
seller_type = st.selectbox('Seller', ['Individual', 'Dealer', 'Trustmark Dealer'])
country = st.selectbox('Country of origin', ['India', 'Czech Republic', 'Japan', 'South Korea', 'United States', 'France', 'Geramany', 'Italy'])

predict_button = st.button('Predict')

if predict_button == True:
    
    # loading scalers and model
    Scaler = joblib.load('Scaler.pkl')
    Target_scaler = joblib.load('Target_scaler.pkl')
    Model = joblib.load('Model.sav')
    

    # encoding categorical variables
    make_map = {'Maruti': 7, 'Skoda': 10, 'Honda': 4, 'Hyundai': 5, 'Toyota': 12, 'Ford': 3, 'Renault': 9, 'Mahindra': 6, 'Tata': 11, 'Chevrolet': 0, 'Datsun': 1, 'Volkswagen': 13, 'Nissan': 8,'Fiat': 2}
    model_map = {'Swift': 65, 'Rapid': 53, 'City': 12, 'i20': 84, 'Xcent': 79, 'Wagon': 77,'800': 0, 'Etios': 27, 'Figo': 29, 'Duster': 17, 'Zen': 81, 'KUV': 40, 'Ertiga': 25,'Alto': 3, 'Verito': 73, 'WR-V': 76, 'SX4': 57, 'Tigor': 71, 'Baleno': 6, 'Enjoy': 24,'Omni': 50, 'Vitara': 75, 'Verna': 74, 'GO': 31, 'Safari': 58, 'Innova': 38, 'Amaze': 4,'Ciaz': 11, 'Jazz': 39, 'Manza': 43, 'i10': 86, 'Ameo': 5, 'Indica': 36, 'Vento': 72,'EcoSport': 19, 'Celerio': 10, 'Polo': 52, 'Eeco': 21, 'Scorpio': 61, 'Freestyle': 30,'Indigo': 37, 'Corolla': 14, 'Terrano': 68, 'Creta': 15, 'KWID': 41, 'Santro': 60,'Elantra': 22, 'Nexon': 48, 'Ritz': 55, 'Grand': 33, 'Zest': 82, 'Getz': 32, 'Elite': 23,'Brio': 9, 'Sunny': 64, 'Micra': 44, 'XUV500': 78, 'Accent': 2, 'Ignis': 35, 'Tiago': 70,'Thar': 69, 'Sumo': 63, 'New': 47, 'Bolero': 8, 'Beat': 7, 'A-Star': 1, 'Nano': 46,'EON': 18, 'RediGO': 54, 'Fiesta': 28, 'Civic': 13, 'Sail': 59, 'Ecosport': 20, 'TUV': 66,'Xylo': 80, 'Grande': 34, 'S-Cross': 56, 'Tavera': 67, 'Linea': 42, 'Esteem': 26, 'Octavia': 49,'Spark': 62, 'Optra': 51, 'Mobilio': 45, 'Cruze': 16}            
    transmission_map = {'Manual': 1, 'Automatic': 0}
    fuel_map = {'Diesel': 1, 'Petrol': 3, 'LPG': 2, 'CNG': 0}
    Owner_map = {'First Owner': 0, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 1}
    seller_type_map = {'Individual': 1, 'Dealer': 0, 'Trustmark Dealer': 2}
    country_map = {'India': 3, 'Czech Republic': 0, 'Japan': 5, 'South Korea': 6, 'United States': 7, 'France': 1, 'Germany': 2, 'Italy': 4}


    make_encoded = make_map[make]
    model_encoded = model_map[model]
    transmission_encoded= transmission_map[transmission]
    fuel_encoded = fuel_map[fuel]
    owner_encoded = Owner_map[owner]
    seller_type_encoded = seller_type_map[seller_type]
    country_encoded = country_map[country]


    input_data = np.array([[make_encoded, model_encoded, year, km_driven, transmission_encoded, fuel_encoded, mileage, engine, max_power, Nm, rpm, seats, owner_encoded, seller_type_encoded, country_encoded]])
    
    scaled_input_data = Scaler.transform(input_data)

    scaled_prediction = Model.predict(scaled_input_data)
    
    output = Target_scaler.inverse_transform(scaled_prediction.reshape(-1,1))
    st.success(output)
