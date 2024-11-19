import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib as plt
import plotly.express as px

df = pd.read_csv('titanic.csv')

st.title('Titanic data exploration')
st.write('App to view Titanic data')

st.sidebar.header('Options')
sex_filter = st.sidebar.multiselect('Select gender', 
                                    options=df['sex'].unique(), 
                                    default=df['sex'].unique())

class_filter= st.sidebar.multiselect('Select class', 
                                     options= df['pclass'].unique(), 
                                     default=df['pclass'].unique())

embarked_filter = st.sidebar.multiselect('Select embarked town',
                                         options=df['embark_town'].unique(),
                                         default=df['embark_town'].unique())


filtered_df = df[
    (df['sex'].isin(sex_filter)) & 
    (df['pclass'].isin(class_filter)) & 
    (df['embark_town'].isin(embarked_filter))]

# Viz 1: Survival countplot
st.subheader("Survival Count")
survival_count = filtered_df['survived'].value_counts()
st.bar_chart(survival_count.rename(index={0: 'Did not survive', 1: 'Survived'}))


# Viz 2: Survival by gender
st.subheader("Survival Rate by Gender")

gender_survival = filtered_df.groupby('sex')['survived'].mean().reset_index()

fig =px.pie(gender_survival, values = 'survived', names='sex')
st.plotly_chart(fig)

age_bins = pd.cut(filtered_df['age'].dropna(), bins=10)
hist_data = age_bins.value_counts()

st.subheader("Age distribution")
hist_df = pd.DataFrame({
    'Age Range': [f"{int(bin.left)}-{int(bin.right)}" for bin in hist_data.index],  # Bin ranges as labels
    'Count': hist_data.values
})

st.bar_chart(hist_df.set_index('Age Range'))