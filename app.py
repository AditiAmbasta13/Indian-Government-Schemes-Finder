import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib  # Use joblib to load the model and encoders

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    excel_file = 'government_schemes.xlsx'
    schemes_df = pd.read_excel(excel_file, sheet_name=0)
    income_df = pd.read_excel(excel_file, sheet_name=1)
    return schemes_df, income_df

# Load model and encoders with joblib
model = joblib.load('scheme_recommendation_model.joblib')
le_gender = joblib.load('le_gender.joblib')
le_age_group = joblib.load('le_age_group.joblib')
le_income_category = joblib.load('le_income_category.joblib')
le_primary_benefit = joblib.load('le_primary_benefit.joblib')

schemes_df, income_df = load_data()

st.title("Indian Government Schemes Finder")

# User input section and prediction
with st.expander("User Information", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Enter Your Name")
        age = st.number_input("Enter Your Age", min_value=1, max_value=120, step=1)
        occupation = st.text_input("Enter Your Occupation")
    with col2:
        gender = st.selectbox("Select Gender", ["Female", "Both", "Male"])
        age_group = st.selectbox("Select Age Group", sorted(schemes_df['Age_Group'].unique()))
        income_category = st.selectbox("Select Income Category", income_df['Income_Category'].tolist())

# Encode inputs for prediction
gender_encoded = le_gender.transform([gender])[0]
age_group_encoded = le_age_group.transform([age_group])[0]
income_category_encoded = le_income_category.transform([income_category])[0]

# Make prediction
prediction = model.predict([[gender_encoded, age_group_encoded, income_category_encoded]])
predicted_benefit = le_primary_benefit.inverse_transform(prediction)[0]

# Display AI-Recommended Primary Benefit outside of the input section
st.header(f"Recommended Primary Benefit: {predicted_benefit}")

# Filter schemes based on AI-predicted Primary Benefit
filtered_df = schemes_df[schemes_df['Primary_Benefit'] == predicted_benefit]

# Display results with side-by-side table and graph
col1, col2 = st.columns(2)

with col1:
    st.subheader("Suitable Schemes")
    if not filtered_df.empty:
        st.dataframe(filtered_df[['Scheme_Name', 'Provider', 'Primary_Benefit', 'Secondary_Benefit', 'Launch_Year', 'Status']], height=300)
    else:
        st.write("No schemes found for the predicted primary benefit.")

with col2:
    st.subheader("Top 5 Schemes Based on Recommendation")
    # Plot a bar chart for the top 5 scheme names
    if not filtered_df.empty:
        top_5_schemes = filtered_df['Scheme_Name'].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top_5_schemes.index, top_5_schemes.values, color='skyblue')
        ax.set_title("Top 5 Schemes", fontsize=16)
        ax.set_xlabel("Frequency", fontsize=12)
        ax.set_ylabel("Scheme Name", fontsize=12)
        plt.gca().invert_yaxis()  # Invert y-axis to display the most frequent scheme at the top
        st.pyplot(fig)
    else:
        st.write("No schemes available to display in the chart.")
