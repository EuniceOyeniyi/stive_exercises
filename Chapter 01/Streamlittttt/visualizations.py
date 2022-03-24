import streamlit as st
import numpy as np
import pandas as pd

st.title('My web app')
st.write('Table')

data = pd.DataFrame(np.random.randn(10,20), columns=('col %d' % i  for i in range(20)))
st.write(data)

st.write('Line chart')
st.line_chart(data)

st.write('Area chart')
st.area_chart(data)

st.write('Histogram')
st.bar_chart(data)


st.write('Map viz')
df = pd.DataFrame(
    np.random.randn(1000,2)/[60,60] + [36.66, 121.6 ],
    columns=['latitude', 'longitude'])
st.map(df)

