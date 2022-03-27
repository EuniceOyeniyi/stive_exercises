import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px

###

# full csv
df = pd.read_csv("https://raw.githubusercontent.com/chriswmann/datasets/master/500_Person_Gender_Height_Weight_Index.csv")
st.title("Body Mass Index of different individuals")

img = Image.open("BMI2.jpg")
st.image(img)



# 1 graph

# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.header("Gender view of each BMI")
# df.groupby('Gender')['Index'].value_counts().plot(kind='bar')
# st.pyplot()


# # 2 graph
# data_select = st.sidebar.selectbox('select your dataset', ('Male', 'Female'))
# st.write(data_select)
# plt.scatter(df['Weight'], df['Height'], c=df.Index )
# st.pyplot()
plot = px.scatter(data_frame = df, x =df['Weight'], y=df['Height'] )
st.plotly_chart(plot)


# # 3 graph
# plt.figure(figsize=(15,8))
fig,ax = plt.subplots()
sns.scatterplot(x='Weight', y='Height', hue='Index', data=df ,palette="deep",ax=ax)
plt.scatter(df.Weight.mean(),df.Height.mean(),color='r', marker='o')
# plt.show()
st.pyplot(fig)


# # 4 graph
# male_height = df[df.Gender == 'Male'].Height
# male_weight = df[df.Gender == 'Male'].Weight

# plt.scatter(male_weight, male_height, c=male_weight.index)
# plt.scatter(df.Weight.mean(),df.Height.mean(),color='r', marker='o')
# st.pyplot()

# # 5 graph

# female_height = df[df.Gender == 'Female'].Height
# female_weight = df[df.Gender == 'Female'].Weight

# plt.scatter(female_weight, female_height, c=female_weight.index)
# plt.scatter(df.Weight.mean(),df.Height.mean(),color='r', marker='o')
# st.pyplot()


# #6 graph

# plt.hist(male_height)
# plt.grid(True, linestyle='--', linewidth = 1)
# plt.title('Males height', color='White')
# st.pyplot()

# plt.hist(male_weight)
# plt.grid(True, linestyle='--', linewidth = 1)
# plt.title('Males weight', color='White')
# st.pyplot()

# # 7 graph









