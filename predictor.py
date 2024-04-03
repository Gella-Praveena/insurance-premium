# Import required libraries
import streamlit as st
import pandas as pd
import plotly.express as px  # Import Plotly
import plotly.graph_objects as go
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


st.markdown(
    """
     <style>
            body {
                background-color: #121212;
                font-family: 'Lucida Bright', sans-serif;
                font-size: 50px;
                color: black;
            }

            h1, h2, h3, h4, h5, h6 {
                font-family: 'Lucida Bright', Cascadia Code, sans-serif;
                color: #BFCFE7;
            }

            .stApp {
                background-color: #212121;
                color: white;
                max-width: 100%;
                margin: 0;
            }

            
        </style>
    """,
    unsafe_allow_html=True,
)


st.markdown('<link href="styles.css" rel="stylesheet">', unsafe_allow_html=True)

st.sidebar.title("CREATING A WEALTH OF HAPPINESS")

st.sidebar.image("health insurance.png", width=280)

nav = st.sidebar.radio("Navigation", ["About", "Predict","Sales","reference"])


df = pd.read_csv('insurance.csv')


#ABOUT SECTION
# If 'About' is selected in the navigation sidebar
if nav == "About":
    st.title("Health Insurance Premium Predictor")
    
    st.write("Welcome to the Health Insurance Premium Predictor app! "
             "This app uses a machine learning model to estimate health insurance premiums based on your details.")
    
    
    st.image('f2.jpg', width=800) 

    st.title("What is Health Insurance and How to Select a Good Policy?") 

    st.write("A health insurance policy is a contract between you and your insurance company. Under this contract,you pay a certain amount (premium) on a regular basis to your insurance company. In turn, the insurance company agrees to help you with the medical expenses that you may incur due to unexpected illness or injury. These expenses typically include doctor consultation fees, diagnostic tests, medicine bills, ambulance fares, hospitalisation bills and more as per your policy terms and conditions.")
    st.write("You can ease this medical burden with a good health insurance policy, where your insurance company will pay for your healthcare expenses. It is important to remember that the amount of money the insurance company will pay for your medical bills depends on the terms and conditions of your health insurance plan. Some plans will require you to pay a certain amount of money out of pocket before the insurance company starts paying for your healthcare expenses.")

    st.title("Benefits of Buying a Health Insurance Policy") 

    # Create two columns for Mitigating Medical Inflation
    col1, col2, = st.columns(2)

    # Add content to each column
    col1.header("Mitigating Medical Inflation")
    col1.write("Medical expenses are rising and a severe injury can easily cost you lakhs of rupees in hospital bills. A health insurance plan not only helps you to deal with the financial liability caused by a medical emergency but also ensures timely treatments.")

    col2.header("     ")
    image_col2 = "medical.webp"  
    col2.image(image_col2, caption='', use_column_width=True)

    # Create two columns for Tax Savings*
    col1, col2, = st.columns(2)

    # Add content to each column
    col2.header("     ")
    image_col1 = "tax.png"  
    col1.image(image_col1, caption='', use_column_width=True)

    col2.header("Tax Savings*")
    col2.write("Under Section 80D of the Income Tax Act, 1961, you can claim tax deductions of up to Rs. 1 lakh per year for your health insurance premiums. These deduction are based on your age and that of your family members.")

    # Create two columns for 
    col1, col2, = st.columns(2)

    # Add content to each column Cashless Claims
    col1.header("Cashless Claims")
    col1.write("All health insurance providers have established tie-ups with numerous network hospitals that allow you to leverage the benefit of cashless claims. This smoothens the process of getting medical assistance in case of an emergency.")

    col2.header("     ")
    image_col2 = "cashless.webp"  
    col2.image(image_col2, caption='', use_column_width=True)


#INSURANCE PREMIUM PREDICTION SECTION
# If 'Predict' is selected in the navigation sidebar
    
df.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)

df.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)

df.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

x = df.drop(columns='charges', axis=1)
y = df['charges']

rfr = RandomForestRegressor()


rfr.fit(x, y)


if nav == "Predict":
    
    st.title("Enter Details for Premium Prediction")

    
    full_name = st.text_input("Full Name:")
    date_of_birth = st.date_input("Date of Birth:", min_value=datetime.date(1950, 1, 1), max_value=datetime.date(2020, 1, 1), value=datetime.date(2020, 1, 1))
    age = st.number_input("Age: ", step=1, min_value=0)
    sex = st.radio("Sex", ("Male", "Female"))
       
    height_units = st.radio("Height Units", ("Feet/Inches", "Meters"))
    if height_units == "Feet/Inches":
        height_ft = st.number_input("Height (Feet):", step=1, min_value=0)
        height_in = st.number_input("Height (Inches):", step=1, min_value=0)
       
        height_meters = (height_ft * 0.3048) + (height_in * 0.0254)
    else:
        # height_meters = st.number_input("Height (Meters):", min_value=0)
        height_meters = st.number_input("Height (Meters):", min_value=0.0, step=0.01)


    weight_units = st.radio("Weight Units", ("Pounds", "Kilograms"))
    if weight_units == "Pounds":
        weight = st.number_input("Weight (Pounds):", min_value=0)
        
        weight_kg = weight * 0.453592
    else:
        weight_kg = st.number_input("Weight (Kilograms):", min_value=0)

    pre_existing_conditions = st.text_input("Any Pre-existing Health Conditions (e.g., diabetes, hypertension)")
    bmi = st.number_input("BMI: ", min_value=0)
    children = st.number_input("Number of children: ", step=1, min_value=0)
    smoke = st.radio("Do you smoke", ("Yes", "No"))
    region = st.selectbox('Region', ('SouthEast', 'SouthWest', 'NorthEast', 'NorthWest'))
    
    coverage_amount = st.number_input("Desired Coverage Amount or Policy Limit ($):", min_value=0)
   
    deductible_applicable = st.checkbox("Deductible Applicable")

    if deductible_applicable:
        deductible = st.number_input("Deductible Amount ($):", min_value=0)
    else:
        deductible = 0 


    has_previous_insurance = st.checkbox("Do you have previous health insurance?")
    if has_previous_insurance:
        current_provider = st.text_input("Current Health Insurance Provider:")
        policy_expiration_date = st.date_input("Current Policy Expiration Date:")


    income = st.number_input("Annual Income ($):", min_value=0)

  
    health_score = st.number_input("Health Condition Score (0-10):", step=0.1, min_value=0.0, max_value=10.0)

    
    health_plan = st.selectbox('Health Plan Type', ('Basic', 'Standard', 'Premium'))
    additional_comments = st.text_area("Additional Comments or Notes","Please provide any additional information or specific requests that might affect your health insurance rate.")
    consent_acknowledge = st.checkbox("I acknowledge that the information provided is accurate to the best of my knowledge. I consent to use this information to calculate a health insurance rate of interest.")

    
    # Convert categorical input fields to numeric values using if-else statements
    if sex == "Male":
        s = 0
    if sex == "Female":
        s = 1
    if smoke == "Yes":
        sm = 0
    if smoke == "No":
        sm = 1
    if region == "SouthEast":
        reg = 0
    if region == "SouthWest":
        reg = 1
    if region == "NorthEast":
        reg = 2
    if region == "NorthWest":
        reg = 3

    
    if st.button("Predict Premium"):
    
     if consent_acknowledge:
        
        predicted_premium = rfr.predict([[age, s, bmi, children, sm, reg]])

        
        st.subheader("Predicted Premium")
       

        
        insurance_data = {
            'Category': ['Age', 'BMI', 'Children', 'Smoker', 'Region'],
            'Value': [age, bmi, children, sm, reg]
        }
        fig = px.pie(insurance_data, names='Category', values='Value', title='Input Feature Distribution')

       
        st.plotly_chart(fig)

        
        result_text = f"<div style='font-size: 24px; color:#87CEEB; font-weight: bold;'>The estimated health insurance premium for {full_name} is: "
        result_text += f"<span style='color:#90EE90; font-weight: bold; font-size: 26px;'>Rs {predicted_premium[0]:.2f}</span></div>"



        st.write(result_text, unsafe_allow_html=True)
     else:
        st.warning("Please acknowledge and consent to use the provided information to calculate the health insurance rate of interest.")



# Load data from a CSV file into a DataFrame
def load_data():
    return pd.read_csv('sales.csv')

# Call the load_data function to load the data
data = load_data()

# Example data loading and splitting
# X contains the features (independent variables)
X = data.drop(columns=['Sales'], axis=1)
# y contains the target variable (dependent variable)
y = data['Sales']

# One-hot encode categorical variables 'sex' and 'region'
X = pd.get_dummies(X, columns=['sex', 'region'], drop_first=True)

# Create a RandomForestRegressor model and fit it with the data
model = RandomForestRegressor()
model.fit(X, y)

# Initialize sales_feature_inputs and features_used_for_training
sales_feature_inputs = {}
features_used_for_training = []


if nav == "Sales":
    # Display a title for the sales forecasting analysis
    st.title("Sales Forecasting Analysis")

    # Display a line chart for historical sales data using Plotly Express
    fig_sales = px.line(data, x=data.index, y='Sales', title='Historical Sales Data')
    st.plotly_chart(fig_sales)

    # Extract features used for training from the one-hot encoded DataFrame
    features_used_for_training = X.columns.tolist()

    # Sales forecasting form in the sidebar
    st.sidebar.subheader('Sales Forecasting')

    # Loop to collect feature inputs
    for feature in X.columns:
        # Check if the column is numeric and not one-hot encoded
        if pd.api.types.is_numeric_dtype(X[feature]) and not feature.startswith(('sex_', 'region_')):
            # Calculate minimum, maximum, and mean values for the numeric feature
            min_value = float(X[feature].min())
            max_value = float(X[feature].max())
            mean_value = float(X[feature].mean())

            # Collect user input for the feature using a number input widget in the sidebar
            sales_feature_inputs[feature] = st.sidebar.number_input(
                f'Enter {feature} for Sales Forecast',
                min_value=min_value,
                max_value=max_value,
                value=mean_value
            )

    # Create a DataFrame from the collected user inputs
    sales_feature_df = pd.DataFrame([sales_feature_inputs])

    # Ensure that the DataFrame has the same columns as features_used_for_training
    sales_feature_df = sales_feature_df.reindex(columns=features_used_for_training, fill_value=0)

    # Convert data types to match the model input
    sales_feature_df = sales_feature_df.astype(float)

    # Check if the DataFrame is empty before making predictions
    if not sales_feature_df.empty:
        # Make sales predictions using the trained RandomForestRegressor model
        sales_prediction = model.predict(sales_feature_df)

        # Display the predicted sales result in the sidebar
        st.sidebar.subheader('Sales Forecast Result')
        st.sidebar.write(f"Predicted Sales: {sales_prediction[0]}")
    else:
        st.sidebar.subheader('Sales Forecast Result')
        st.sidebar.write("Please enter values for sales forecasting.")





# REFERENCE SECTION

# If 'Reference' is selected in the navigation sidebar
if nav == "reference":
    st.title("Best Health Insurance Companies in India ")
    st.write("There are currently 30 insurance companies in India that offer reliable health insurance plans. Out"
             "of these, 25 are general insurance companies and 5 are standalone health insurance companies."
             "All these companies have unique features of their own and cater to different requirements of customers.")
    
    st.title("Top Health Insurance Companies Claim Settlement Ratio ")
    st.image('table.png', width=800) 
    st.image('table2.png', width=800)

    st.title("Health insurance plans")
    st.title("Mediclaim Policy ")
    st.write("Medical inflation is on the rise. Lifestyle diseases are increasing too. Getting hospitalized on an"
             "emergency basis without having health cover can exhaust your savings in no time. Buying a"
             "mediclaim policy can provide you with the required financial assistance in case of hospitalization."
             "The mediclaim policy provides coverage against medical expenses that you may incur during the"
             "policy period. With a valid mediclaim, you can also receive tax benefits under section 80D of the"
             "Income Tax Act, 1961.")
    
    st.title("Health Insurance Plans for Family")
    st.write("Health Insurance Plans for Family is a kind of medical insurance that covers the entire family under"
             "a fixed sum insured. Under family health care insurance, you can cover two or more family"
             "members for several healthcare expenses including in-patient hospitalisation expenses, pre-"
             "hospitalisation, post-hospitalisation expenses, daycare expenses, and a few more.")
    
    st.title("Health Insurance Plans for Senior Citizens")
    st.write("Senior citizen health insurance plans are specially designed to cover people aged over 60 years."
             "Available on an individual or family floater basis, these plans offer the utmost protection to elderly "
             "people in their old age. One can also avail of tax benefits on the premium paid for these plans.")
    
    st.title("Women Health Insurance")
    st.write("Women health insurance plans cover the women for specific healthcare requirements which not"
             "all regular health insurance plans cover them for. These include maternity expenses, newborn"
             "baby coverage, critical illness cover, etc. These plans offer women financial independence.")
    
    st.title("Children Health Insurance")
    st.write("Health insurance plans for children offer coverage to children for various healthcare expenses"
             "arising due to the child getting ill, sick, or contracting any disease. The coverage can be availed as"
             "per the sum insured chosen and the premium paid.")

        