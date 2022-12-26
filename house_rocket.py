#================ Libraries ===============
import geopandas
import streamlit as st
import pandas as pd
import numpy as np
import folium

from datetime import datetime, time
from datetime import date

from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

import plotly.express as px

#================ Settings ================
st.set_page_config (layout='wide')

#================ Extraction ================

@st.cache (allow_output_mutation=True)
def get_data (path):
    data = pd.read_csv (path)

    return data

@st.cache (allow_output_mutation=True)
def get_geofile (url):
    geofile = geopandas.read_file (url)

    return geofile

#================ Transformation ================

def set_attribute (data):
    data['date'] = pd.to_datetime(data['date']).dt.date
    data['price_m2'] = data['price'] / data['sqft_lot']

    return data

def data_overview (data):
    data2 = data

    #------------ filters ------------

    f_zipcode = st.sidebar.multiselect ('Select Zipcode', data['zipcode'].sort_values().unique())
    f_atributes = st.sidebar.multiselect ('Select Columns', data.columns)

    st.header ('Data Overview')

    if (f_zipcode != []) & (f_atributes != []):  
        data = data.loc[data['zipcode'].isin (f_zipcode), f_atributes]
        data2 = data2.loc[data2['zipcode'].isin(f_zipcode), :]

    elif (f_zipcode != []) & (f_atributes == []):
        data = data.loc[data['zipcode'].isin (f_zipcode), :]
        data2 = data2.loc[data2['zipcode'].isin(f_zipcode), :]

    elif (f_zipcode == []) & (f_atributes != []):
        data = data.loc[:, f_atributes]

    else:
        data = data.copy()

    st.dataframe (data)

    df1 = data2[['id', 'zipcode']].groupby ('zipcode').count().reset_index()
    df2 = data2[['price', 'zipcode']].groupby ('zipcode').mean().reset_index()
    df3 = data2[['sqft_living', 'zipcode']].groupby ('zipcode').mean().reset_index()
    df4 = data2[['price_m2' , 'zipcode']].groupby ('zipcode').mean().reset_index()

    m1 = pd.merge (df1, df2, on = 'zipcode', how = 'inner')
    m2 = pd.merge (m1, df3, on = 'zipcode', how = 'inner')
    df = pd.merge (m2, df4, on = 'zipcode', how = 'inner')

    c1, c2 = st.columns ((1, 1))

    c1.subheader('Average Values')
    df.columns = ['zipcode', 'Total Houses', 'Average Price', 'Average Sqft Living', 'Average Price m2']
    c1.dataframe (df, width= 800)

    num_atributes = data.select_dtypes (include = ['int64', 'float64'])
    media = pd.DataFrame (num_atributes.apply (np.mean))
    mediana = pd.DataFrame (num_atributes.apply (np.median))
    std = pd.DataFrame (num_atributes.apply (np.std))
    max_ = pd.DataFrame (num_atributes.apply (np.max))
    min_ = pd.DataFrame (num_atributes.apply (np.min))

    df = pd.concat ([media, mediana, std, max_, min_], axis=1).reset_index()
    df.columns = ['Atributes', 'Media', 'Median', 'Standart', 'Max', 'Min']

    c2.subheader('Descriptive Analisis')
    c2.dataframe (df, width=800)

    return data, data2

def region_overview (data2, geofile):
    st.title ('Region Overview')
    c1, c2 = st.columns((1, 1))

    # ------------ portfolio density -----------------
    
    c1.header ('Density Portfolio')
    df = data2.sample (10)

    density_map = folium.Map (location = [data2['lat'].mean(), data2['long'].mean()], default_zoom_start = 15)
    marker_cluster = MarkerCluster().add_to(density_map)

    for name, row in df.iterrows():
        folium.Marker([row['lat'], row['long']], popup='Sold R${0} on {1}.Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built {5}'.format(row['price'], row['date'], row['sqft_lot'], row['bedrooms'], row['bathrooms'], row['yr_built'])).add_to(marker_cluster)

    with c1:
        folium_static (density_map)

    # ------------ price density -----------------

    c2.header ('Price Density')
    df = data2[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    geofile = geofile[geofile['ZIP'].isin(df['ZIP'].to_list())]

    region_price_map = folium.Map (location = [data2['lat'].mean(), data2['long'].mean()], default_zoom_start = 15)
    region_price_map.choropleth (data = df, geo_data = geofile, columns = ['ZIP', 'PRICE'], key_on = 'feature.properties.ZIP', fill_color = 'YlOrRd', fill_opacity = 0.7, line_opacity = 0.2, legend_name = 'AVG PRICE')

    with c2:
        folium_static (region_price_map)
    
    return None

def price_graph (data2):
    # filters
    min_date = data2['date'].min()
    max_date = data2['date'].max()
    min_year_built = int (data2['yr_built'].min())
    max_year_built = int (data2['yr_built'].max())
    f_yr_min, f_yr_max = st.sidebar.select_slider ('Year Date', options = data2['yr_built'].sort_values(), value = (min_year_built, max_year_built))
    f_date_min, f_date_max = st.sidebar.select_slider ('Range Date', options = data2['date'].sort_values(), value = (min_date, max_date))

    c3, c4 = st.columns((1, 1))

    #------------ price by year built --------------

    data2 = data2.loc[(data2['yr_built'] > f_yr_min) & (data2['yr_built'] < f_yr_max)]
    df = data2[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    fig = px.line (df, x = 'yr_built', y = 'price')

    c3.subheader('Price by year built')
    c3.plotly_chart (fig, user_container_width = True)

    #------------ average day price -------------

    data2 = data2.loc[(data2['date'] > f_date_min) & (data2['date'] < f_date_max)]
    df = data2[['date', 'price']].groupby('date').mean().reset_index()

    fig = px.line (df, x = 'date', y = 'price')

    c4.subheader ('Average day price')
    c4.plotly_chart (fig, user_container_width = True)

    # ------------- price distribution ---------

    st.header ('Price Distribution')

    min_price = int (data2['price'].min())
    max_price = int (data2['price'].max())

    st.sidebar.subheader ('Distribution Attributes')
    f_price = st.sidebar.slider('Price', min_price, max_price, max_price)
    data2 = data2.loc[data2['price'] < f_price]

    fig = px.histogram(data2, x = 'price', nbins = 50)
    st.plotly_chart(fig, user_container_width = True)

    return data2

def houses_attributes (data2):
    c1, c2 = st.columns ((1, 1))

    #------------- houses per bedrooms -------------

    c1.subheader ('Distribution by bedrooms')
    f_bedrooms = st.sidebar.selectbox ('Max Bedrooms', data2['bedrooms'].sort_values(ascending = False).unique())
    data2 = data2.loc[data2['bedrooms'] < f_bedrooms]
    fig = px.histogram (data2, x = 'bedrooms', nbins = 19)
    c1.plotly_chart (fig, user_container_width = True)

    #------------- houses per bathrooms -------------

    c2.subheader ('Distribution by bathrooms')
    f_bathrooms = st.sidebar.selectbox ('Max Bathrooms', data2['bathrooms'].sort_values( ascending = False).unique())
    data2 = data2.loc[data2['bathrooms'] < f_bathrooms]
    fig = px.histogram (data2, x = 'bathrooms', nbins = 19)
    c2.plotly_chart (fig, user_container_width = True)

    c1, c2 = st.columns ((1, 1))

    # -------------houses per floors -------------

    c1.subheader ('Distribution by floors')
    f_floors = st.sidebar.selectbox ('Max Floors', data2['floors'].sort_values(ascending = False).unique())
    data2 = data2.loc[data2['floors'] < f_floors]
    fig = px.histogram (data2, x = 'floors', nbins = 10)
    c1.plotly_chart (fig, user_container_width = True)

    #------------- waterfront houses -------------

    c2.subheader ('Distribution by waterview')
    f_waterfront = st.sidebar.checkbox('Waterview houses')
    if f_waterfront:
        data2 = data2.loc[data2['waterfront'] == 1]
    else:
        data2 = data2.copy()
    fig = px.histogram (data2, x = 'waterfront', nbins = 10)
    c2.plotly_chart (fig, user_container_width = True)

    return data2
#Load

def buy_options (data):

#------------- buy portfolio -------------

    st.header ('Recommended houses to buy')
    price_median = data[['price', 'zipcode']].groupby('zipcode').median().reset_index()
    price_median = pd.merge(data, price_median, on='zipcode', how='inner')
    price_median = price_median.rename(columns = {'price_y': 'price_median', 'price_x': 'price'})
    buy_options = price_median.loc[price_median['price'] < price_median['price_median']]
    buy_options = buy_options.loc[buy_options['condition'] >= 3]

    st.dataframe(buy_options)

    return None

if __name__ == '__main__':
    #load data
    path = 'kc_house_data.csv'
    geofile = get_geofile('https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson')
    data = get_data (path)

    #transform data
    data = set_attribute (data)
    data, data2 = data_overview (data)
    region_overview (data2, geofile)
    data2 = price_graph (data2)
    data2 = houses_attributes (data2)
    buy_options (data)
