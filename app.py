import pickle
import pandas as pd
import streamlit as st
# Set basic page configuration
st.set_page_config(
    page_title="Churn_Prediction_Model",
    page_icon="osha",
    layout="centered"
)

# Load the model
model = pickle.load(open("model.pkl", "rb"))

# Create page header
st.markdown("#Customer Churn Prediction Model")

# Take inputs from user
age = st.text_input("Customer Age:")
gender = st.selectbox("Customer Gender", ['Male', 'Female'])
location = st.selectbox("Customer Location", ['Houston', 'Los Angeles', 'Miami', 'Chicago', 'New York'])
subscription_length_months = st.text_input("Subscription Months:")
monthly_bill = st.text_input("Monthly Bill ($):")
total_used_gb = st.text_input("Total Usage (GB):")

# Create an array of all these inputs
features = [{
    'Age': age,
    'Gender': gender,
    'Location': location,
    'Subscription_Length_Months': subscription_length_months,
    'Monthly_Bill': monthly_bill,
    'Total_Usage_GB': total_used_gb
}]

# Convert it to pandas DataFrame before passing it to the model
features_df = pd.DataFrame(features)

if total_used_gb:
    output = model.predict(features_df)
    if output == 1:
        st.error("Churn Warning: Customer might exit.")
        st.write("With a '1' on the radar, the model suggests a noteworthy probability that this customer might be contemplating a departure.")
    elif output == 0:
        st.success("Churn Resistant: Customer is Loyal.")
        st.write("With a '0' on the scene, it's like the model is giving a thumbs-up to this customer's staying power. The chances of them leaving? Well, let's just say they're safely tucked in the 'unlikely' zone.")
