import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👤",
    layout="wide", 
    initial_sidebar_state="expanded"
)

try:
    model = joblib.load(open('Attrition_Model.pkl', 'rb'))
    scaler = joblib.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: Model or Scaler file not found. Make sure 'Attrition_Model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop() # Stop execution if files are missing
except Exception as e:
    st.error(f"An error occurred loading the model/scaler: {e}")
    st.stop()

st.markdown("""
<style>
    /* Base font */
    html, body, [class*="st-"] {
        font-family: 'Times New Roman', Times, serif;
    }

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        background-color: #022873;
    }

    /* Title style */
    h1 {
        color: #037057;
        text-align: center;
        text-shadow: 2px 2px 4px #cccccc;
    }

    /* Section headers */
    h2 {
        color: #15803D; /* Dark Green */
        border-bottom: 2px solid #15803D;
        padding-bottom: 5px;
        margin-top: 2rem;
    }

    /* Sidebar styling */
    .stSidebar {
        background-color: #035910; /* Light Cyan/Blue */
        padding: 15px;
    }
    .stSidebar h2 {
        color: #c98e02; /* Dark Gray for sidebar header */
        border-bottom: none;
    }

    /* Input labels in sidebar */
    .stSidebar .stWidget > label {
        color: #1F2937; /* Slightly darker gray for labels */
        font-weight: bold;
    }

    /* Button styling */
    .stButton>button {
        background-image: linear-gradient(to right, #1E3A8A , #1E40AF); /* Blue gradient */
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 8px;
        font-size: 1.1em;
        font-weight: bold;
        box-shadow: 3px 3px 6px #b0b0b0;
        transition: all 0.3s ease;
        width: 100%; /* Make button fill width */
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-image: linear-gradient(to right, #1E40AF , #1D4ED8); /* Slightly lighter blue on hover */
        box-shadow: 5px 5px 8px #999999;
        transform: translateY(-2px); /* Slight lift on hover */
    }

    /* Prediction result area */
    .prediction-area {
        padding: 1.5rem;
        margin-top: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5em; /* Larger font for result */
        font-weight: bold;
    }
    .prediction-yes {
        background-color: #FEE2E2; /* Light Red */
        border: 2px solid #DC2626; /* Red border */
        color: #DC2626; /* Red text */
    }
    .prediction-no {
        background-color: #DCFCE7; /* Light Green */
        border: 2px solid #16A34A; /* Green border */
        color: #16A34A; /* Green text */
    }

     /* Description text */
    .description-text {
         color: #000103; /* Dark Gray */
         font-size: 1.1em;
         line-height: 1.6;
         background-color:#088080;
         padding: 1rem;
         border-radius: 8px;
         box-shadow: 2px 2px 5px #cccccc;
         margin-bottom: 2rem;
    }

</style>
""", unsafe_allow_html=True)

st.title("👤 Employee Attrition Prediction")

st.markdown("## Understanding Employee Attrition")
st.markdown("""
<div class="description-text">
Employee attrition, often referred to as employee turnover, is the gradual reduction in the number of employees
through retirement, resignation, or death. High attrition rates can be costly for organizations due to expenses
related to recruitment, onboarding, training, and lost productivity. Understanding the factors that contribute
to attrition allows businesses to implement strategies to improve employee retention, foster a better work
environment, and maintain a stable, experienced workforce. This tool uses a Machine Learning model
(Logistic Regression) trained on various employee attributes to predict the likelihood of an employee leaving the company.
Enter the employee's details in the sidebar to get a prediction.
</div>
""", unsafe_allow_html=True)

st.sidebar.header("📊 Employee Details Input")

def user_input_features():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Personal & General Info")
    Age = st.sidebar.slider('Age', 18, 60, 35)
    Gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    MaritalStatus = st.sidebar.selectbox('Marital Status', ('Married', 'Single', 'Divorced'))
    Department = st.sidebar.selectbox('Department', ('Sales', 'Research & Development', 'Human Resources'))

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Job & Role Details")
    JobRole = st.sidebar.selectbox('Job Role', (
        'Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
        'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'
    ))
    JobLevel = st.sidebar.slider('Job Level', 1, 5, 2)
    JobInvolvement = st.sidebar.slider('Job Involvement (1:Low - 4:Very High)', 1, 4, 3)
    JobSatisfaction = st.sidebar.slider('Job Satisfaction (1:Low - 4:Very High)', 1, 4, 3)
    EnvironmentSatisfaction = st.sidebar.slider('Environment Satisfaction (1:Low - 4:Very High)', 1, 4, 3)
    RelationshipSatisfaction = st.sidebar.slider('Relationship Satisfaction (1:Low - 4:Very High)', 1, 4, 3)
    WorkLifeBalance = st.sidebar.slider('Work Life Balance (1:Bad - 4:Best)', 1, 4, 3)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Compensation & Performance")
    DailyRate = st.sidebar.slider('Daily Rate ($)', 100, 1500, 800)
    HourlyRate = st.sidebar.slider('Hourly Rate ($)', 30, 100, 65)
    MonthlyIncome = st.sidebar.slider('Monthly Income ($)', 1000, 20000, 5000, step=100)
    MonthlyRate = st.sidebar.slider('Monthly Rate ($)', 2000, 27000, 14000, step=100)
    PercentSalaryHike = st.sidebar.slider('Percent Salary Hike (%)', 11, 25, 15)
    PerformanceRating = st.sidebar.selectbox('Performance Rating (3:High, 4:Outstanding)', (3, 4))
    StockOptionLevel = st.sidebar.slider('Stock Option Level', 0, 3, 1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Career & Experience")
    Education = st.sidebar.selectbox('Education Level (1:Below College - 5:Doctorate)', (1, 2, 3, 4, 5))
    EducationField = st.sidebar.selectbox('Education Field', ('Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other')) # Encoded: LS=1, Med=3, Mark=2, Tech=5, HR=0, Oth=4 (Assumption!)
    NumCompaniesWorked = st.sidebar.slider('Number of Companies Worked At', 0, 9, 1)
    TotalWorkingYears = st.sidebar.slider('Total Working Years', 0, 40, 10)
    TrainingTimesLastYear = st.sidebar.slider('Training Times Last Year', 0, 6, 3)
    YearsAtCompany = st.sidebar.slider('Years At Company', 0, 40, 5)
    YearsInCurrentRole = st.sidebar.slider('Years in Current Role', 0, 18, 3)
    YearsSinceLastPromotion = st.sidebar.slider('Years Since Last Promotion', 0, 15, 1)
    YearsWithCurrManager = st.sidebar.slider('Years With Current Manager', 0, 17, 3)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Travel & Other")
    BusinessTravel = st.sidebar.selectbox('Business Travel Frequency', ('Non-Travel', 'Travel_Rarely', 'Travel_Frequently'))
    DistanceFromHome = st.sidebar.slider('Distance From Home (miles)', 1, 30, 5)
    OverTime = st.sidebar.selectbox('OverTime', ('Yes', 'No'))

    gender_map = {'Male': 1, 'Female': 0}
    overtime_map = {'Yes': 1, 'No': 0}
    travel_map = {'Non-Travel': 0, 'Travel_Rarely': 2, 'Travel_Frequently': 1}
    dept_map = {'Sales': 2, 'Research & Development': 1, 'Human Resources': 0}
    edufield_map = {'Life Sciences': 1, 'Medical': 3, 'Marketing': 2, 'Technical Degree': 5, 'Human Resources': 0, 'Other': 4} 
    marital_map = {'Married': 1, 'Single': 2, 'Divorced': 0}
    perf_rating_map = {3: 0, 4: 1}
    jobrole_map = {
        'Sales Executive': 8, 'Research Scientist': 6, 'Laboratory Technician': 2, 'Manufacturing Director': 3,
        'Healthcare Representative': 1, 'Manager': 4, 'Sales Representative': 7, 'Research Director': 5, 'Human Resources': 0
    }

    data = {
        'Age': Age,
        'BusinessTravel': travel_map[BusinessTravel],
        'DailyRate': DailyRate,
        'Department': dept_map[Department],
        'DistanceFromHome': DistanceFromHome,
        'Education': Education,
        'EducationField': edufield_map[EducationField],
        'EnvironmentSatisfaction': EnvironmentSatisfaction,
        'Gender': gender_map[Gender],
        'HourlyRate': HourlyRate,
        'JobInvolvement': JobInvolvement,
        'JobLevel': JobLevel,
        'JobRole': jobrole_map[JobRole],
        'JobSatisfaction': JobSatisfaction,
        'MaritalStatus': marital_map[MaritalStatus],
        'MonthlyIncome': MonthlyIncome,
        'MonthlyRate': MonthlyRate,
        'NumCompaniesWorked': NumCompaniesWorked,
        'OverTime': overtime_map[OverTime],
        'PercentSalaryHike': PercentSalaryHike,
        'PerformanceRating': perf_rating_map[PerformanceRating],
        'RelationshipSatisfaction': RelationshipSatisfaction,
        'StockOptionLevel': StockOptionLevel,
        'TotalWorkingYears': TotalWorkingYears,
        'TrainingTimesLastYear': TrainingTimesLastYear,
        'WorkLifeBalance': WorkLifeBalance,
        'YearsAtCompany': YearsAtCompany,
        'YearsInCurrentRole': YearsInCurrentRole,
        'YearsSinceLastPromotion': YearsSinceLastPromotion,
        'YearsWithCurrManager': YearsWithCurrManager
    }
    
    feature_order = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
                     'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
                     'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole',
                     'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
                     'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
                     'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
                     'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                     'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                     'YearsWithCurrManager']

    features_df = pd.DataFrame([data])
    features_df = features_df[feature_order]

    return features_df

input_df = user_input_features()

st.markdown("## Prediction Result")

# Button
if st.button('Predict Attrition'):
    try:
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Display result
        st.markdown("---")
        if prediction[0] == 1:
            st.markdown(f"""
            <div class="prediction-area prediction-yes">
                Prediction: <span style="font-weight:bold;">Employee is LIKELY to leave</span> (Attrition = Yes)<br>
                Confidence Score (Probability of Leaving): {prediction_proba[0][1]:.2f}
            </div>
            """, unsafe_allow_html=True)
            st.warning("Consider implementing retention strategies for employees with similar profiles.")
        else:
            st.markdown(f"""
            <div class="prediction-area prediction-no">
                Prediction: <span style="font-weight:bold;">Employee is UNLIKELY to leave</span> (Attrition = No)<br>
                Confidence Score (Probability of Staying): {prediction_proba[0][0]:.2f}
            </div>
            """, unsafe_allow_html=True)
            st.success("The model predicts this employee profile is likely to be retained.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure all inputs are valid and the model/scaler were loaded correctly.")

else:
    st.info("Click the 'Predict Attrition' button after adjusting the employee details in the sidebar.")

st.markdown("---")
st.markdown("_Disclaimer: This prediction is based on a statistical model and should be used as a guide, not a definitive outcome._")
