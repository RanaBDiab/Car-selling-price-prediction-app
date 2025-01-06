import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler


# configuration
st.set_page_config(layout= 'wide')
st.header('Car Selling Price Prediction app')


# side bar
# app info

st.sidebar.image('car_app_vec_02.jpg')
st.sidebar.write('This project was made as part of the Kayfa scholarship in Data science with Imperious University of America.')
st.sidebar.write('In aworld where car prices are always rising, the purpose of this app is to help:')
st.sidebar.write('- Sellers: learn the market value of the car they plan to sell')
st.sidebar.write('- Buyers: learn the fair value of the car they plan on buying ')
st.sidebar.write('On this app you can:')
st.sidebar.write('1- Explore the data set used in this project')
st.sidebar.write('2- Test the AI model to predict the selling price of your car')

# contact info
st.sidebar.title('Contact')
st.sidebar.write("[LinkedIn](https://www.linkedin.com/in/ranabdiab/)")
st.sidebar.write("[Github](https://github.com/RanaBDiab)")
st.sidebar.write('ranadiab26@gmail.com')


# Building Tabs
dashboard, model_tab = st.tabs(["Explore Data", "Test AI Model"])
    
# Building dashboard
with dashboard:
        # importing data: 
        df= pd.read_csv('reg_car_selling_final.csv')




        country = st.selectbox('Country', options=["All"] + ['India', 'Czech Republic', 'Japan', 'South Korea', 
                       'United States', 'France', 'Geramany', 'Italy'])
    

        if country != "All":
             df = df[df['country'] == country]
      

        # row 2
        # cards

        card1, card2, card3, card4 = st.columns(4)
        card1.metric('Average mileage', df['mileage'].mean().round(2))
        card2.metric('Average max power', df['max_power'].mean().round(2))
        card3.metric('Average rpm', df['rpm'].mean().round(2))
        card4.metric('Average Nm/kgm', df['Nm/kgm'].mean().round(2))


        # row 3
        # owner and seller type
        column_f, column_g = st.columns((5,5))

        with column_f:
            owner_data = df['owner'].value_counts().tolist()
            owner_unique = df['owner'].unique()
            owner_fig = px.pie(values= owner_data, names= owner_unique, title = 'Owner distribution')
            owner_fig.update_layout(xaxis_showgrid= False, yaxis_showgrid = False,
              legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5))
            st.plotly_chart(owner_fig, use_container_width=True)
    

        with column_g:
            seller_data = df['seller_type'].value_counts().tolist()
            seller_unique = df['seller_type'].unique()
            seller_fig = px.pie(values= seller_data, names= seller_unique, title = 'Seller type distribution')
            seller_fig.update_layout(xaxis_showgrid= False, yaxis_showgrid = False,
              legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5))
            st.plotly_chart(seller_fig, use_container_width=True)
        
        
      

        # row 4
        # transmission and fuel

        column_h, column_i = st.columns((5,5))

        with column_h:
            transmission_data = df['transmission'].value_counts().tolist()
            transmission_unique = df['transmission'].unique()
            transmission_fig = px.pie(values= transmission_data, names= transmission_unique, title = 'Transmission distribution')
            transmission_fig.update_layout(xaxis_showgrid= False, yaxis_showgrid = False,
              legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5))
            st.plotly_chart(transmission_fig, use_container_width=True)


        with column_i:
            fuel_data = df['fuel'].value_counts().tolist()
            fuel_unique = df['fuel'].unique()
            fuel_fig = px.pie(values= transmission_data, names= transmission_unique, title = 'Transmission distribution')
            fuel_fig.update_layout(xaxis_showgrid= False, yaxis_showgrid = False,
              legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5))
            st.plotly_chart(fuel_fig, use_container_width=True)



    # row 5
    # make and price per kilometer
    
        column_j, column_k = st.columns((5,5))


        with column_j:
            make_data = df['make'].value_counts().tolist()
            make_labels = df['make'].unique()
            make_fig = px.bar(df, x= make_labels, y= make_data, title= 'Make distribution' )
            make_fig.update_layout(xaxis_title="make", yaxis_title="number of cars")
            st.plotly_chart(make_fig, use_container_width=True)

        

        with column_k:
            price_distance_data= df.groupby('km_driven')['selling_price'].mean()
            price_km_fig = px.scatter(price_distance_data, title='Average selling price by distance travelled')
            price_km_fig.update_layout(xaxis_title="distance travelled", yaxis_title="average price")
            st.plotly_chart(price_km_fig, use_container_width=True)


    
        # row 6
        # selling price per year
        
        price_year_data = df.groupby('year')['selling_price'].sum()
        price_year_fig= px.line(price_year_data, 
                     x=price_year_data.index, 
                     y=price_year_data.values, 
                     labels={'value': 'average selling price', 'year': 'year'}
                     )
        price_year_fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
        st.plotly_chart(price_year_fig, use_container_width=True)


st.write('')
st.write('')
st.write('')

# building model tab

with model_tab:
     make = st.selectbox('Make', ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun','Volkswagen', 'Nissan', 'Fiat'])
model = st.selectbox('Model', ['Swift', 'Rapid', 'City', 'i20', 'Xcent', 'Wagon', '800', 
       'Etios', 'Figo', 'Duster', 'Zen', 'KUV', 'Ertiga', 'Alto', 'Verito', 'WR-V', 'SX4',
       'Tigor', 'Baleno', 'Enjoy', 'Omni', 'Vitara', 'verna', 'GO', 'Safari', 'Innova',
       'Amaze', 'Ciaz', 'jazz', 'Manza', 'i10', 'Ameo', 'Indica', 'Vento', 'EcoSport', 'Celerio',
       'Polo', 'Eeco', 'Scorpio', 'Freestyle', 'Indigo', 'Corolla', 'Terrano', 'Creta', 'KWID', 
       'Santro', 'Elantra', 'Nexon', 'Ritz', 'Grand', 'Zest', 'Getz', 'Elite', 'Brio', 'Sunny',
       'Micra', 'XUV500', 'Accent', 'Ignis', 'Tiago', 'Thar', 'Sumo', 'New', 'Bolero', 'Beat',
       'A-Star', 'Nano', 'EON', 'RediGO', 'Fiesta', 'Civic', 'Sail', 'Ecosport', 'TUV', 'Xylo',
       'Grande', 'S-Cross', 'Tavera', 'Linea', 'Esteem', 'Octavia', 'Spark', 'Optra', 'Mobilio', 'Cruze'] )
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
country = st.selectbox('Country of origin', ['India', 'Czech Republic', 'Japan', 'South Korea', 
                       'United States', 'France', 'Geramany', 'Italy'])

predict_button = st.button('Predict')

if predict_button == True:
    
    # loading scalers and model
    Scaler = joblib.load('Scaler.pkl')
    Target_scaler = joblib.load('Target_scaler.pkl')
    Model = joblib.load('Model.sav')
    

    # encoding categorical variables
    make_map = {'Maruti': 7, 'Skoda': 10, 'Honda': 4, 'Hyundai': 5, 'Toyota': 12, 
                'Ford': 3, 'Renault': 9, 'Mahindra': 6, 'Tata': 11, 'Chevrolet': 0, 
                'Datsun': 1, 'Volkswagen': 13, 'Nissan': 8,'Fiat': 2}
    model_map = {'Swift': 65, 'Rapid': 53, 'City': 12, 'i20': 84, 'Xcent': 79, 'Wagon': 77,
                '800': 0, 'Etios': 27, 'Figo': 29, 'Duster': 17, 'Zen': 81, 'KUV': 40, 'Ertiga': 25,
                'Alto': 3, 'Verito': 73, 'WR-V': 76, 'SX4': 57, 'Tigor': 71, 'Baleno': 6, 'Enjoy': 24,
                'Omni': 50, 'Vitara': 75, 'Verna': 74, 'GO': 31, 'Safari': 58, 'Innova': 38, 'Amaze': 4,
                'Ciaz': 11, 'Jazz': 39, 'Manza': 43, 'i10': 86, 'Ameo': 5, 'Indica': 36, 'Vento': 72,
                'EcoSport': 19, 'Celerio': 10, 'Polo': 52, 'Eeco': 21, 'Scorpio': 61, 'Freestyle': 30,
                'Indigo': 37, 'Corolla': 14, 'Terrano': 68, 'Creta': 15, 'KWID': 41, 'Santro': 60, 
                'Elantra': 22, 'Nexon': 48, 'Ritz': 55, 'Grand': 33, 'Zest': 82, 'Getz': 32, 'Elite': 23,
                'Brio': 9, 'Sunny': 64, 'Micra': 44, 'XUV500': 78, 'Accent': 2, 'Ignis': 35, 'Tiago': 70,
                'Thar': 69, 'Sumo': 63, 'New': 47, 'Bolero': 8, 'Beat': 7, 'A-Star': 1, 'Nano': 46, 
                'EON': 18, 'RediGO': 54, 'Fiesta': 28, 'Civic': 13, 'Sail': 59, 'Ecosport': 20, 'TUV': 66,
                'Xylo': 80, 'Grande': 34, 'S-Cross': 56, 'Tavera': 67, 'Linea': 42, 'Esteem': 26, 'Octavia': 49,
                'Spark': 62, 'Optra': 51, 'Mobilio': 45, 'Cruze': 16}            
    transmission_map = {'Manual': 1, 'Automatic': 0}
    fuel_map = {'Diesel': 1, 'Petrol': 3, 'LPG': 2, 'CNG': 0}
    Owner_map = {'First Owner': 0, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 1}
    seller_type_map = {'Individual': 1, 'Dealer': 0, 'Trustmark Dealer': 2}
    country_map = {'India': 3, 'Czech Republic': 0, 'Japan': 5, 'South Korea': 6, 'United States': 7,
                   'France': 1, 'Germany': 2, 'Italy': 4}


    make_encoded = make_map[make]
    model_encoded = model_map[model]
    transmission_encoded= transmission_map[transmission]
    fuel_encoded = fuel_map[fuel]
    owner_encoded = Owner_map[owner]
    seller_type_encoded = seller_type_map[seller_type]
    country_encoded = country_map[country]


    input_data = np.array([[make_encoded, model_encoded, year, km_driven, transmission_encoded, 
                            fuel_encoded, mileage, engine, max_power, Nm, rpm,
                             seats, owner_encoded, seller_type_encoded, country_encoded]])
    
    scaled_input_data = Scaler.transform(input_data)

    scaled_prediction = Model.predict(scaled_input_data)
    
    output = Target_scaler.inverse_transform(scaled_prediction.reshape(-1,1))
    st.success(output)
