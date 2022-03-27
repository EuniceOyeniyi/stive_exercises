import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from PIL import Image

st.markdown("<h1 style='text-align: center; color: black;'>Looking for the best crime movie to watch this weekend?</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: blue;'>Let's get started with the adventure!!! </h2>", unsafe_allow_html=True)

col1, col2 , col3 = st.columns(3)

with col1:
    st.write('')

with col2:
    image = Image.open('CrimeMovies.jpg')
    st.image(image)


with col3:
    st.write(' ')

df = pd.read_csv("Finaldata.csv")
# fig = plt.figure()
# fig = df[['Release', 'rating']].groupby('Release').mean().plot(kind='bar', title='Average rating of movies per year')
# plt.xlabel('years')
# plt.ylabel('rating')
group_data = df[['Release', 'rating']].groupby('Release').mean()
# group_data = group_data.reset_index()
fig = plt.figure()

group_data.plot.bar()
st.pyplot()


